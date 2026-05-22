const express = require("express");
const cors = require("cors");
const { EventEmitter } = require("events");
const { ChatOpenAI } = require("@langchain/openai");
const { HumanMessage } = require("@langchain/core/messages");

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static("public"));

const emitter = new EventEmitter();
emitter.setMaxListeners(100);

// ── Prompt variation (vLLM 캐시 회피) ────────────────────────────────
// 사용자의 기본 프롬프트에 무작위 컨텍스트를 삽입하여
// 각 요청마다 다른 프롬프트를 생성합니다.
// 구조/길이는 비슷하게 유지하면서 문자열만 다르게 만들어 캐시를 회피합니다.

// ── 대규모 코드베이스 컨텍스트 생성 ─────────────────────────────
// 실사용 시 사용자가 긴 코드베이스를 프롬프트에 포함하는 상황을 모의합니다.
// 다량의 TypeScript/JavaScript 코드를 생성하여 input token 수를 크게 늘립니다.

const CODEBASE_TEMPLATES = {
  types: `
// types/user.ts
export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  createdAt: Date;
  updatedAt: Date;
  profile: UserProfile | null;
  settings: UserSettings;
  metadata: Record<string, unknown>;
}

export type UserRole = 'admin' | 'moderator' | 'user' | 'guest' | 'banned';

export interface UserProfile {
  bio: string;
  avatarUrl: string;
  website: string;
  location: string;
  socialLinks: SocialLink[];
  skills: string[];
}

export interface SocialLink {
  platform: 'twitter' | 'github' | 'linkedin' | 'website';
  url: string;
  label: string;
}

export interface UserSettings {
  theme: 'light' | 'dark' | 'system';
  language: string;
  timezone: string;
  emailNotifications: boolean;
  pushNotifications: boolean;
  twoFactorEnabled: boolean;
  sessionTimeout: number;
}

// types/document.ts
export interface Document {
  id: string;
  title: string;
  content: string;
  authorId: string;
  createdAt: Date;
  updatedAt: Date;
  version: number;
  tags: string[];
  status: DocumentStatus;
  parentId: string | null;
  children: Document[];
  metadata: DocumentMetadata;
}

export type DocumentStatus = 'draft' | 'published' | 'archived' | 'deleted';

export interface DocumentMetadata {
  wordCount: number;
  readingTime: number;
  language: string;
  categories: string[];
  relatedIds: string[];
}

// types/analytics.ts
export interface AnalyticsEvent {
  id: string;
  type: EventType;
  userId: string;
  sessionId: string;
  timestamp: Date;
  properties: Record<string, unknown>;
  source: string;
  platform: 'web' | 'ios' | 'android' | 'api';
}

export type EventType =
  | 'page_view'
  | 'click'
  | 'form_submit'
  | 'error'
  | 'api_call'
  | 'file_upload'
  | 'search'
  | 'filter_apply'
  | 'export'
  | 'import';

export interface AnalyticsSummary {
  period: { start: Date; end: Date };
  totalEvents: number;
  uniqueUsers: number;
  eventsByType: Record<EventType, number>;
  eventsByPlatform: Record<string, number>;
  topPages: Array<{ path: string; views: number }>;
  errorRate: number;
  avgSessionDuration: number;
}`,

  service: `
// services/authService.ts
import { User, UserRole } from '../types/user';
import { ValidationError, AuthenticationError } from '../errors';
import { hashPassword, verifyPassword, generateToken } from '../utils/crypto';
import { rateLimiter } from '../middleware/rateLimiter';
import { logger } from '../utils/logger';
import { redisClient } from '../config/redis';

export class AuthService {
  private readonly sessionStore: Map<string, UserSession>;
  private readonly refreshTokens: Map<string, RefreshTokenData>;
  private readonly otpCodes: Map<string, OTPData>;

  constructor(private readonly userRepo: UserRepository) {
    this.sessionStore = new Map();
    this.refreshTokens = new Map();
    this.otpCodes = new Map();
  }

  async login(email: string, password: string, options?: LoginOptions): Promise<LoginResult> {
    const startTime = Date.now();
    const ip = options?.ipAddress || 'unknown';

    // Rate limiting check
    const limitResult = await rateLimiter.check(ip, 'login', 5, 60000);
    if (!limitResult.allowed) {
      logger.warn(\`Rate limit exceeded for login from IP: \${ip}\`);
      throw new AuthenticationError('너무 많은 로그인 시도입니다. 잠시 후 다시 시도해주세요.');
    }

    // Find user by email
    const user = await this.userRepo.findByEmail(email);
    if (!user) {
      logger.info(\`Login attempt for non-existent email: \${email} from \${ip}\`);
      // Use constant-time comparison to prevent timing attacks
      await verifyPassword(password, '$2b$10$dummyHashForTimingAttackPrevention1234567890');
      throw new AuthenticationError('이메일 또는 비밀번호가 올바르지 않습니다.');
    }

    // Check if account is active
    if (user.status === 'banned') {
      logger.warn(\`Login attempt for banned user: \${user.id} from \${ip}\`);
      throw new AuthenticationError('이 계정은 정지되었습니다.');
    }

    if (user.status === 'suspended') {
      throw new AuthenticationError('이 계정은 일시 정지되었습니다. 관리자에게 문의하세요.');
    }

    // Verify password
    const isPasswordValid = await verifyPassword(password, user.passwordHash);
    if (!isPasswordValid) {
      logger.warn(\`Invalid password for user: \${user.id} from \${ip}\`);
      const attempts = await this.getFailedLoginAttempts(user.id);
      const maxAttempts = 5;

      if (attempts >= maxAttempts) {
        await this.lockAccount(user.id, 30 * 60 * 1000); // 30 minutes
        throw new AuthenticationError(
          \`계정이 잠겼습니다. \${30}분 후 다시 시도해주세요.\`
        );
      }

      await this.incrementFailedAttempts(user.id);
      const remaining = maxAttempts - attempts - 1;
      throw new AuthenticationError(
        \`이메일 또는 비밀번호가 올바르지 않습니다. 나머지 시도 횟수: \${remaining}\`
      );
    }

    // Check two-factor authentication
    if (user.twoFactorEnabled && options?.twoFactorCode) {
      const isValid = await this.verifyTwoFactorCode(user.id, options.twoFactorCode);
      if (!isValid) {
        throw new AuthenticationError('인증 코드가 올바르지 않습니다.');
      }
    }

    // Reset failed attempts on successful login
    await this.resetFailedAttempts(user.id);

    // Create session
    const session = await this.createSession(user, { ip, userAgent: options?.userAgent });

    // Generate tokens
    const accessToken = generateToken({ userId: user.id, role: user.role }, '15m');
    const refreshToken = generateToken({ userId: user.id, type: 'refresh' }, '7d');

    // Store refresh token
    this.refreshTokens.set(refreshToken, {
      userId: user.id,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
      ipAddress: ip,
    });

    const duration = Date.now() - startTime;
    logger.info(\`Successful login for user \${user.id} in \${duration}ms\`);

    return {
      user: this.sanitizeUser(user),
      session,
      accessToken,
      refreshToken,
      twoFactorRequired: user.twoFactorEnabled && !options?.twoFactorCode,
    };
  }

  async refreshToken(refreshToken: string): Promise<AccessTokenResult> {
    const tokenData = this.refreshTokens.get(refreshToken);
    if (!tokenData) {
      throw new AuthenticationError('무효화된 리프레시 토큰입니다.');
    }

    if (tokenData.expiresAt < new Date()) {
      this.refreshTokens.delete(refreshToken);
      throw new AuthenticationError('만료된 리프레시 토큰입니다.');
    }

    const user = await this.userRepo.findById(tokenData.userId);
    if (!user || user.status !== 'active') {
      this.refreshTokens.delete(refreshToken);
      throw new AuthenticationError('사용자 계정을 찾을 수 없습니다.');
    }

    const newAccessToken = generateToken({ userId: user.id, role: user.role }, '15m');

    return {
      accessToken: newAccessToken,
      expiresAt: new Date(Date.now() + 15 * 60 * 1000),
    };
  }

  async logout(sessionId: string): Promise<void> {
    this.sessionStore.delete(sessionId);
    logger.info(\`Session \${sessionId} terminated\`);
  }

  async createOTP(userId: string): Promise<OTPResult> {
    const code = Math.floor(100000 + Math.random() * 900000).toString();
    const expiresAt = new Date(Date.now() + 5 * 60 * 1000);

    this.otpCodes.set(userId, { code, expiresAt, createdAt: new Date() });

    // In production, send via email/SMS
    logger.info(\`OTP generated for user \${userId}\`);

    return {
      expiresAt,
      expiresInSeconds: 300,
    };
  }

  private async createSession(user: User, options: SessionOptions): Promise<UserSession> {
    const sessionId = generateToken({ random: crypto.randomUUID() }, '1h');
    const session: UserSession = {
      id: sessionId,
      userId: user.id,
      createdAt: new Date(),
      lastActive: new Date(),
      ipAddress: options.ipAddress,
      userAgent: options.userAgent || 'unknown',
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
    };

    this.sessionStore.set(sessionId, session);

    // Cache in Redis for distributed sessions
    await redisClient.set(
      \`session:\${sessionId}\`,
      JSON.stringify(session),
      'EX',
      86400
    );

    return session;
  }

  private sanitizeUser(user: User): SanitizedUser {
    const { passwordHash, twoFactorSecret, ...safeUser } = user;
    return safeUser as unknown as SanitizedUser;
  }
}`,

  controller: `
// controllers/documentController.ts
import { Request, Response } from 'express';
import { Document, DocumentStatus } from '../types/document';
import { DocumentService } from '../services/documentService';
import { ValidationError, NotFoundError, PermissionError } from '../errors';
import { paginate, parseSearchParams } from '../utils/pagination';
import { logger } from '../utils/logger';
import { cache } from '../middleware/cache';
import { validateRequestBody } from '../middleware/validation';

export class DocumentController {
  constructor(private readonly documentService: DocumentService) {}

  @cache({ ttl: 300, keyPrefix: 'docs:list' })
  async index(req: Request, res: Response) {
    const { page, limit, sort, order, search, tags, status, authorId } =
      parseSearchParams(req.query);

    try {
      const filter = {
        status: status as DocumentStatus | undefined,
        authorId,
        tags: tags?.split(','),
        searchQuery: search,
      };

      const result = await this.documentService.findAll(filter, {
        page,
        limit,
        sort: sort || 'updatedAt',
        order: order || 'desc',
      });

      const transformed = result.items.map(doc =>
        this.transformDocument(doc)
      );

      return res.json({
        data: transformed,
        pagination: {
          page: result.page,
          limit: result.limit,
          total: result.total,
          totalPages: result.totalPages,
          hasNext: result.page < result.totalPages,
          hasPrev: result.page > 1,
        },
        meta: {
          requestDuration: result.duration,
          cacheHit: res.getHeader('X-Cache') === 'HIT',
        },
      });
    } catch (error) {
      logger.error('Failed to fetch documents:', error);
      return res.status(500).json({
        error: '서버 오류가 발생했습니다.',
        code: 'INTERNAL_ERROR',
      });
    }
  }

  @cache({ ttl: 60, keyPrefix: 'docs:detail' })
  async show(req: Request, res: Response) {
    const { id } = req.params;
    const user = res.locals.user;

    try {
      const document = await this.documentService.findById(id);
      if (!document) {
        return res.status(404).json({
          error: '문서를 찾을 수 없습니다.',
          code: 'DOCUMENT_NOT_FOUND',
        });
      }

      // Check access permissions
      if (!this.documentService.canAccess(document, user)) {
        return res.status(403).json({
          error: '이 문서에 접근할 권한이 없습니다.',
          code: 'ACCESS_DENIED',
        });
      }

      // Increment view count
      await this.documentService.incrementViews(id);

      return res.json({
        data: this.transformDocument(document),
        meta: {
          requestDuration: 0,
          cacheHit: res.getHeader('X-Cache') === 'HIT',
        },
      });
    } catch (error) {
      logger.error(\`Failed to fetch document \${id}:\`, error);
      return res.status(500).json({
        error: '서버 오류가 발생했습니다.',
        code: 'INTERNAL_ERROR',
      });
    }
  }

  async store(req: Request, res: Response) {
    const user = res.locals.user;
    const validatedBody = req.validatedBody;

    try {
      const document = await this.documentService.create({
        ...validatedBody,
        authorId: user.id,
      });

      logger.info(\`Document created: \${document.id} by user \${user.id}\`);

      return res.status(201).json({
        data: this.transformDocument(document),
        message: '문서가 성공적으로 생성되었습니다.',
      });
    } catch (error) {
      if (error instanceof ValidationError) {
        return res.status(400).json({
          error: error.message,
          code: 'VALIDATION_ERROR',
          details: error.details,
        });
      }

      logger.error('Failed to create document:', error);
      return res.status(500).json({
        error: '문서 생성 중 오류가 발생했습니다.',
        code: 'INTERNAL_ERROR',
      });
    }
  }

  async update(req: Request, res: Response) {
    const { id } = req.params;
    const user = res.locals.user;
    const validatedBody = req.validatedBody;

    try {
      const existing = await this.documentService.findById(id);
      if (!existing) {
        return res.status(404).json({
          error: '문서를 찾을 수 없습니다.',
          code: 'DOCUMENT_NOT_FOUND',
        });
      }

      if (!this.documentService.canEdit(existing, user)) {
        return res.status(403).json({
          error: '이 문서를 수정할 권한이 없습니다.',
          code: 'EDIT_DENIED',
        });
      }

      const updated = await this.documentService.update(id, {
        ...validatedBody,
        updatedBy: user.id,
        version: existing.version + 1,
      });

      return res.json({
        data: this.transformDocument(updated),
        message: '문서가 성공적으로 업데이트되었습니다.',
      });
    } catch (error) {
      if (error instanceof ValidationError) {
        return res.status(400).json({
          error: error.message,
          code: 'VALIDATION_ERROR',
          details: error.details,
        });
      }
      logger.error(\`Failed to update document \${id}:\`, error);
      return res.status(500).json({
        error: '문서 업데이트 중 오류가 발생했습니다.',
        code: 'INTERNAL_ERROR',
      });
    }
  }

  async destroy(req: Request, res: Response) {
    const { id } = req.params;
    const user = res.locals.user;

    try {
      const existing = await this.documentService.findById(id);
      if (!existing) {
        return res.status(404).json({
          error: '문서를 찾을 수 없습니다.',
          code: 'DOCUMENT_NOT_FOUND',
        });
      }

      if (!this.documentService.canDelete(existing, user)) {
        return res.status(403).json({
          error: '이 문서를 삭제할 권한이 없습니다.',
          code: 'DELETE_DENIED',
        });
      }

      await this.documentService.softDelete(id, user.id);

      return res.json({
        message: '문서가 성공적으로 삭제되었습니다.',
      });
    } catch (error) {
      logger.error(\`Failed to delete document \${id}:\`, error);
      return res.status(500).json({
        error: '문서 삭제 중 오류가 발생했습니다.',
        code: 'INTERNAL_ERROR',
      });
    }
  }

  private transformDocument(doc: Document): TransformedDocument {
    return {
      id: doc.id,
      title: doc.title,
      summary: doc.content.substring(0, 200),
      status: doc.status,
      tags: doc.tags,
      version: doc.version,
      createdAt: doc.createdAt.toISOString(),
      updatedAt: doc.updatedAt.toISOString(),
      author: {
        id: doc.authorId,
      },
      metadata: doc.metadata,
      children: doc.children.map(c => ({ id: c.id, title: c.title })),
    };
  }
}`,

  repository: `
// repositories/baseRepository.ts
import { Pool, PoolClient } from 'pg';
import { logger } from '../utils/logger';
import { DatabaseError, NotFoundError } from '../errors';

export interface QueryBuilder {
  select(columns?: string[]): QueryBuilder;
  from(table: string): QueryBuilder;
  where(condition: string, params?: any[]): QueryBuilder;
  orderBy(column: string, direction?: 'ASC' | 'DESC'): QueryBuilder;
  limit(count: number): QueryBuilder;
  offset(count: number): QueryBuilder;
  join(
    table: string,
    condition: string,
    type?: 'INNER' | 'LEFT' | 'RIGHT' | 'FULL'
  ): QueryBuilder;
  groupBy(column: string): QueryBuilder;
  having(condition: string, params?: any[]): QueryBuilder;
  count(column?: string): Promise<number>;
  first(): Promise<Record<string, any> | null>;
  all(): Promise<Record<string, any>[]>;
  execute(): Promise<QueryResult>;
  getSQL(): string;
  getParams(): any[];
}

export class BaseQueryBuilder implements QueryBuilder {
  protected sql = 'SELECT ';
  protected columns: string[] = ['*'];
  protected table = '';
  protected whereClauses: string[] = [];
  protected params: any[] = [];
  protected orderClause = '';
  protected limitClause = '';
  protected offsetClause = '';
  protected joinClauses: string[] = [];
  protected groupClause = '';
  protected havingClause = '';
  protected isCount = false;

  select(columns: string[]): this {
    this.columns = columns;
    this.isCount = false;
    return this;
  }

  from(table: string): this {
    this.table = table;
    return this;
  }

  where(condition: string, params: any[] = []): this {
    this.whereClauses.push(condition);
    this.params.push(...params);
    return this;
  }

  orderBy(column: string, direction: 'ASC' | 'DESC' = 'ASC'): this {
    this.orderClause = \`ORDER BY \${column} \${direction}\`;
    return this;
  }

  limit(count: number): this {
    this.limitClause = \`LIMIT \${count}\`;
    return this;
  }

  offset(count: number): this {
    this.offsetClause = \`OFFSET \${count}\`;
    return this;
  }

  join(
    table: string,
    condition: string,
    type: 'INNER' | 'LEFT' | 'RIGHT' | 'FULL' = 'INNER'
  ): this {
    this.joinClauses.push(\`\${type} JOIN \${table} ON \${condition}\`);
    return this;
  }

  groupBy(column: string): this {
    this.groupClause = \`GROUP BY \${column}\`;
    return this;
  }

  having(condition: string, params: any[] = []): this {
    this.havingClause = \`HAVING \${condition}\`;
    this.params.push(...params);
    return this;
  }

  async count(column: string = '*'): Promise<number> {
    this.isCount = true;
    this.columns = [ \`COUNT(\${column}) AS count\` ];
    const result = await this.first();
    return parseInt(result.count, 10);
  }

  getSQL(): string {
    let sql = '';

    if (this.isCount) {
      sql = 'SELECT ';
    } else {
      sql = 'SELECT ';
    }

    sql += this.columns.join(', ');
    sql += \` FROM \${this.table}\`;

    if (this.joinClauses.length > 0) {
      sql += ' ' + this.joinClauses.join(' ');
    }

    if (this.whereClauses.length > 0) {
      sql += ' WHERE ' + this.whereClauses.join(' AND ');
    }

    if (this.groupClause) {
      sql += ' ' + this.groupClause;
    }

    if (this.havingClause) {
      sql += ' ' + this.havingClause;
    }

    if (this.orderClause) {
      sql += ' ' + this.orderClause;
    }

    if (this.limitClause) {
      sql += ' ' + this.limitClause;
    }

    if (this.offsetClause) {
      sql += ' ' + this.offsetClause;
    }

    return sql;
  }

  getParams(): any[] {
    return [...this.params];
  }

  async first(): Promise<Record<string, any> | null> {
    const sql = this.getSQL();
    const params = this.getParams();

    try {
      const client = await Pool.connect();
      try {
        const result = await client.query(sql, params);
        return result.rows[0] || null;
      } finally {
        client.release();
      }
    } catch (error) {
      logger.error('Database query failed:', { sql, params, error });
      throw new DatabaseError('데이터베이스 쿼리 실패', {
        cause: error,
        query: sql,
        params,
      });
    }
  }

  async all(): Promise<Record<string, any>[]> {
    const sql = this.getSQL();
    const params = this.getParams();

    try {
      const client = await Pool.connect();
      try {
        const result = await client.query(sql, params);
        return result.rows;
      } finally {
        client.release();
      }
    } catch (error) {
      logger.error('Database query failed:', { sql, params, error });
      throw new DatabaseError('데이터베이스 쿼리 실패', {
        cause: error,
        query: sql,
        params,
      });
    }
  }

  async execute(): Promise<QueryResult> {
    const sql = this.getSQL();
    const params = this.getParams();

    try {
      const client = await Pool.connect();
      try {
        const result = await client.query(sql, params);
        return {
          rows: result.rows,
          rowCount: result.rowCount,
          command: result.command,
        };
      } finally {
        client.release();
      }
    } catch (error) {
      logger.error('Database execute failed:', { sql, params, error });
      throw new DatabaseError('데이터베이스 실행 실패', {
        cause: error,
        query: sql,
        params,
      });
    }
  }
}

export abstract class BaseRepository<T> {
  protected constructor(
    protected readonly tableName: string,
    protected readonly pool: Pool
  ) {}

  protected buildQuery(): BaseQueryBuilder {
    return new BaseQueryBuilder();
  }

  abstract findById(id: string): Promise<T | null>;
  abstract findAll(options?: FindOptions): Promise<PaginatedResult<T>>;
  abstract create(data: CreateInput<T>): Promise<T>;
  abstract update(id: string, data: UpdateInput<T>): Promise<T>;
  abstract delete(id: string): Promise<boolean>;
}
`,

  middleware: `
// middleware/rateLimiter.ts
import { RateLimitEntry, RateLimitConfig } from '../types/rateLimit';
import { logger } from '../utils/logger';
import { redisClient } from '../config/redis';

export class RateLimiter {
  private localStore: Map<string, RateLimitEntry>;
  private cleanupInterval: NodeJS.Timeout;

  constructor(
    private readonly config: RateLimitConfig = {
      defaultMaxRequests: 100,
      defaultWindowMs: 60000,
      useRedis: process.env.NODE_ENV === 'production',
    }
  ) {
    this.localStore = new Map();
    // Cleanup expired entries every 60 seconds
    this.cleanupInterval = setInterval(() => this.cleanup(), 60000);
  }

  async check(
    identifier: string,
    key: string,
    maxRequests?: number,
    windowMs?: number
  ): Promise<RateLimitResult> {
    const limit = maxRequests ?? this.config.defaultMaxRequests;
    const window = windowMs ?? this.config.defaultWindowMs;
    const fullKey = \`\${identifier}:\${key}\`;
    const now = Date.now();

    if (this.config.useRedis) {
      return this.checkRedis(fullKey, limit, window, now);
    }

    return this.checkLocal(fullKey, limit, window, now);
  }

  private async checkRedis(
    key: string,
    limit: number,
    window: number,
    now: number
  ): Promise<RateLimitResult> {
    try {
      const pipeline = redisClient.pipeline();
      const windowStart = now - window;

      // Remove old entries outside the window
      pipeline.zremrangebyscore(key, 0, windowStart);

      // Count current entries in window
      pipeline.zcard(key);

      // Add current request
      pipeline.zadd(key, now, \`\${now}:\${Math.random()}\`);

      // Set expiry
      pipeline.expire(key, Math.ceil(window / 1000));

      const results = await pipeline.exec();
      const currentCount = (results[1] as any)[1];

      const remaining = Math.max(0, limit - currentCount - 1);
      const resetTime = now + window;

      return {
        allowed: currentCount < limit,
        limit,
        remaining,
        resetTime,
        retryAfter: currentCount >= limit ? Math.ceil(window / 1000) : 0,
      };
    } catch (error) {
      logger.error('Redis rate limit check failed:', error);
      // Fallback to local on Redis failure
      return this.checkLocal(key, limit, window, now);
    }
  }

  private checkLocal(
    key: string,
    limit: number,
    window: number,
    now: number
  ): RateLimitResult {
    const entry = this.localStore.get(key);

    if (!entry || entry.windowStart < now - window) {
      this.localStore.set(key, {
        windowStart: now,
        count: 1,
        timestamps: [now],
      });

      return {
        allowed: true,
        limit,
        remaining: limit - 1,
        resetTime: now + window,
        retryAfter: 0,
      };
    }

    // Filter timestamps within window
    entry.timestamps = entry.timestamps.filter(t => t >= now - window);
    entry.windowStart = now;
    entry.count = entry.timestamps.length;

    const resetTime = now + window;
    const remaining = Math.max(0, limit - entry.count - 1);

    if (entry.count >= limit) {
      return {
        allowed: false,
        limit,
        remaining: 0,
        resetTime,
        retryAfter: Math.ceil((resetTime - now) / 1000),
      };
    }

    entry.timestamps.push(now);
    entry.count++;

    return {
      allowed: true,
      limit,
      remaining: limit - entry.count,
      resetTime,
      retryAfter: 0,
    };
  }

  private cleanup(): void {
    const now = Date.now();
    let cleaned = 0;

    for (const [key, entry] of this.localStore.entries()) {
      if (now - entry.windowStart > 120000) {
        this.localStore.delete(key);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      logger.debug(\`Cleaned \${cleaned} expired rate limit entries\`);
    }
  }

  shutdown(): void {
    clearInterval(this.cleanupInterval);
    this.localStore.clear();
  }
}

// middleware/cache.ts
import { Response } from 'express';
import { redisClient } from '../config/redis';
import { logger } from '../utils/logger';

export interface CacheOptions {
  ttl: number;
  keyPrefix: string;
  skipIfHasParam?: string[];
  varyBy?: string[];
}

export function cache(options: CacheOptions) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const req = args[0];
      const res = args[1];

      // Skip caching if request has certain params
      if (options.skipIfHasParam) {
        for (const param of options.skipIfHasParam) {
          if (req.query[param] || req.params[param]) {
            return originalMethod.apply(this, args);
          }
        }
      }

      const cacheKey = buildCacheKey(req, options);

      try {
        // Try to serve from cache
        const cached = await redisClient.get(cacheKey);
        if (cached) {
          const { data, statusCode, headers } = JSON.parse(cached);
          logger.debug(\`Cache HIT for \${cacheKey}\`);
          res.set('X-Cache', 'HIT');
          if (headers) res.set(headers);
          return res.status(statusCode).json(data);
        }

        // Execute original handler
        const result = await originalMethod.apply(this, args);

        // Cache the response
        if (res.statusCode >= 200 && res.statusCode < 300) {
          const cacheData = JSON.stringify({
            data: res.locals.data,
            statusCode: res.statusCode,
            headers: res.getHeaders(),
          });

          await redisClient.set(cacheKey, cacheData, 'EX', options.ttl);
          res.set('X-Cache', 'MISS');
        }

        return result;
      } catch (error) {
        logger.error(\`Cache middleware error for \${cacheKey}:\`, error);
        return originalMethod.apply(this, args);
      }
    };

    return descriptor;
  };
}

function buildCacheKey(req: Request, options: CacheOptions): string {
  const parts = [options.keyPrefix, req.method, req.originalUrl];

  if (options.varyBy) {
    for (const header of options.varyBy) {
      parts.push(\${header}=\${req.headers[header] || 'none'});
    }
  }

  const raw = parts.join(':');
  // Simple hash to keep keys short
  return 'cache:' + require('crypto').createHash('sha256').update(raw).digest('hex').substring(0, 32);
}
`,

  hook: `
// hooks/useWebSocket.ts
import { useEffect, useRef, useCallback, useState } from 'react';
import { WebSocketClient, Message } from '../lib/websocket';
import { logger } from '../utils/logger';

interface UseWebSocketOptions<T = any> {
  url: string;
  protocols?: string[];
  onMessage?: (event: MessageEvent<T>) => void;
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: ErrorEvent) => void;
  shouldReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  auth_token?: string;
  debug?: boolean;
}

export function useWebSocket<T = any>(options: UseWebSocketOptions<T>) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [lastMessage, setLastMessage] = useState<Message<T> | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [error, setError] = useState<Error | null>(null);

  const clientRef = useRef<WebSocketClient | null>(null);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimerRef = useRef<NodeJS.Timeout | null>(null);
  const messageQueueRef = useRef<T[]>([]);
  const isMountedRef = useRef(true);

  const maxAttempts = options.maxReconnectAttempts ?? 5;
  const reconnectInterval = options.reconnectInterval ?? 3000;
  const heartbeatInterval = options.heartbeatInterval ?? 30000;

  const connect = useCallback(() => {
    if (!isMountedRef.current) return;
    if (clientRef.current?.readyState === WebSocket.OPEN) return;

    setIsConnecting(true);
    setError(null);

    try {
      const client = new WebSocketClient(options.url, {
        protocols: options.protocols,
        auth_token: options.auth_token,
        debug: options.debug,
      });

      client.onOpen(() => {
        if (!isMountedRef.current) return;
        setIsConnected(true);
        setIsConnecting(false);
        setReconnectAttempts(0);

        // Start heartbeat
        startHeartbeat();

        // Flush queued messages
        const queued = [...messageQueueRef.current];
        messageQueueRef.current = [];
        queued.forEach(msg => client.send(msg));

        options.onOpen?.();
        logger.info('WebSocket connected');
      });

      client.onClose((event) => {
        if (!isMountedRef.current) return;
        setIsConnected(false);
        setIsConnecting(false);
        stopHeartbeat();

        logger.info(\`WebSocket closed: code=\${event.code} reason=\${event.reason}\`);

        if (options.shouldReconnect !== false && reconnectAttempts < maxAttempts) {
          scheduleReconnect();
        }

        options.onClose?.(event);
      });

      client.onError((event) => {
        if (!isMountedRef.current) return;
        const err = new Error(\`WebSocket error: \${event.message}\`);
        setError(err);
        setIsConnected(false);
        setIsConnecting(false);

        options.onError?.(event);
        logger.error('WebSocket error:', event);
      });

      client.onMessage((event) => {
        if (!isMountedRef.current) return;

        let data: T;
        try {
          data = JSON.parse(event.data) as T;
        } catch {
          data = event.data as unknown as T;
        }

        const message: Message<T> = {
          id: crypto.randomUUID(),
          data,
          timestamp: new Date(),
          type: (data as any).type || 'unknown',
        };

        setLastMessage(message);
        options.onMessage?.(event as unknown as MessageEvent<T>);
      });

      clientRef.current = client;

    } catch (err) {
      logger.error('Failed to create WebSocket:', err);
      setError(err as Error);
      setIsConnecting(false);
    }
  }, [options.url, options.shouldReconnect]);

  const disconnect = useCallback((code: number = 1000, reason = '') => {
    stopHeartbeat();
    clearReconnectTimer();

    if (clientRef.current) {
      clientRef.current.close(code, reason);
      clientRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const send = useCallback((data: T) => {
    if (!clientRef.current?.readyState === WebSocket.OPEN) {
      messageQueueRef.current.push(data);
      logger.warn('Message queued (not connected)');
      return false;
    }

    try {
      clientRef.current.send(data);
      return true;
    } catch (err) {
      logger.error('Failed to send message:', err);
      return false;
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current) return;

    reconnectTimerRef.current = setTimeout(() => {
      if (!isMountedRef.current) return;
      setReconnectAttempts(prev => prev + 1);
      reconnectTimerRef.current = null;
      connect();
    }, reconnectInterval * (reconnectAttempts + 1));
  }, [connect, reconnectInterval, reconnectAttempts]);

  const startHeartbeat = useCallback(() => {
    stopHeartbeat();
    heartbeatTimerRef.current = setInterval(() => {
      if (clientRef.current?.readyState === WebSocket.OPEN) {
        clientRef.current.ping();
      }
    }, heartbeatInterval);
  }, [heartbeatInterval]);

  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
  }, []);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    connect();

    return () => {
      isMountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    isConnecting,
    lastMessage,
    reconnectAttempts,
    error,
    connect,
    disconnect,
    send,
  };
}
`,

  component: `
// components/DataGrid/DataGrid.tsx
import React, { useMemo, useCallback, useState, useEffect } from 'react';
import { useGridState } from '../../hooks/useGridState';
import { useSortState } from '../../hooks/useSortState';
import { useFilterState } from '../../hooks/useFilterState';
import { ColumnDef, GridRow, GridCellValue } from '../../types/grid';
import { SortDirection, FilterOperator } from '../../types/common';
import * as S from './DataGrid.styles';

export interface DataGridProps<T extends GridRow> {
  columns: ColumnDef<T>[];
  data: T[];
  loading?: boolean;
  error?: Error | null;
  onRowClick?: (row: T) => void;
  onSelectionChange?: (selectedRows: T[]) => void;
  rowHeight?: number;
  virtualized?: boolean;
  selectable?: boolean;
  sortable?: boolean;
  filterable?: boolean;
  resizable?: boolean;
  striped?: boolean;
  dense?: boolean;
  emptyMessage?: string;
  className?: string;
  style?: React.CSSProperties;
}

export function DataGrid<T extends GridRow>(props: DataGridProps<T>) {
  const {
    columns,
    data,
    loading = false,
    error = null,
    onRowClick,
    onSelectionChange,
    rowHeight = 40,
    virtualized = true,
    selectable = false,
    sortable = true,
    filterable = false,
    resizable = false,
    striped = true,
    dense = false,
    emptyMessage = '데이터가 없습니다.',
    className = '',
    style,
  } = props;

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [columnWidths, setColumnWidths] = useState<Record<string, number>>({});
  const [resizingColumn, setResizingColumn] = useState<string | null>(null);

  const gridState = useGridState({
    columns,
    data,
    virtualized,
    rowHeight: dense ? rowHeight - 8 : rowHeight,
  });

  const { sortedData, sortConfig, handleSort } = useSortState({
    data,
    enabled: sortable,
    defaultSort: { column: columns[0]?.accessor, direction: SortDirection.ASC },
  });

  const { filteredData, filterConfig, setFilter } = useFilterState({
    data: sortedData,
    enabled: filterable,
  });

  const displayData = filteredData;

  const isSelected = useCallback(
    (row: T) => selectedIds.has(String(row.id)),
    [selectedIds]
  );

  const toggleRowSelection = useCallback(
    (row: T) => {
      const newSelected = new Set(selectedIds);
      if (newSelected.has(String(row.id))) {
        newSelected.delete(String(row.id));
      } else {
        newSelected.add(String(row.id));
      }
      setSelectedIds(newSelected);
      onSelectionChange?.(
        displayData.filter(r => newSelected.has(String(r.id)))
      );
    },
    [selectedIds, displayData, onSelectionChange]
  );

  const selectAll = useCallback(() => {
    const allIds = new Set(displayData.map(r => String(r.id)));
    setSelectedIds(allIds);
    onSelectionChange?.(displayData);
  }, [displayData, onSelectionChange]);

  const deselectAll = useCallback(() => {
    setSelectedIds(new Set());
    onSelectionChange?.([]);
  }, [onSelectionChange]);

  const handleCellRender = useCallback(
    (row: T, column: ColumnDef<T>) => {
      const value = row[column.accessor as keyof T];

      if (column.render) {
        return column.render(value, row);
      }

      if (typeof value === 'object' && value !== null) {
        return JSON.stringify(value);
      }

      return String(value ?? '');
    },
    [columns]
  );

  const handleColumnResize = useCallback(
    (columnId: string) => (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setResizingColumn(columnId);

      const startX = e.clientX;
      const startWidth = columnWidths[columnId] || 150;

      const onMouseMove = (moveEvent: MouseEvent) => {
        const diff = moveEvent.clientX - startX;
        const newWidth = Math.max(80, startWidth + diff);
        setColumnWidths(prev => ({ ...prev, [columnId]: newWidth }));
      };

      const onMouseUp = () => {
        setResizingColumn(null);
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
      };

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    },
    [columnWidths]
  );

  useEffect(() => {
    setSelectedIds(new Set());
  }, [displayData.length]);

  if (loading) {
    return <S.LoadingOverlay>Loading...</S.LoadingOverlay>;
  }

  if (error) {
    return <S.ErrorContainer>{error.message}</S.ErrorContainer>;
  }

  if (displayData.length === 0) {
    return <S.EmptyState>{emptyMessage}</S.EmptyState>;
  }

  return (
    <S.GridContainer className={className} style={style}>
      <S.HeaderRow>
        {selectable && (
          <S.CheckboxCell>
            <input
              type="checkbox"
              checked={
                selectedIds.size > 0 &&
                selectedIds.size === displayData.length
              }
              onChange={
                selectedIds.size === displayData.length
                  ? deselectAll
                  : selectAll
              }
            />
          </S.CheckboxCell>
        )}
        {columns.map(col => (
          <S.HeaderCell
            key={col.accessor}
            style={{ width: columnWidths[col.accessor] || col.width || 150 }}
            onClick={() => sortable && handleSort(col.accessor)}
            className={sortConfig?.column === col.accessor ? 'sorted' : ''}
          >
            <S.HeaderContent>
              {col.header || col.accessor}
              {sortable && (
                <S.SortIndicator
                  active={sortConfig?.column === col.accessor}
                  direction={sortConfig?.column === col.accessor ? sortConfig.direction : null}
                >
                  ↕
                </S.SortIndicator>
              )}
            </S.HeaderContent>
            {resizable && (
              <S.ResizeHandle onMouseDown={handleColumnResize(col.accessor)} />
            )}
          </S.HeaderCell>
        ))}
      </S.HeaderRow>

      <S.BodyWrapper virtualized={virtualized} rowHeight={dense ? rowHeight - 8 : rowHeight}>
        {displayData.map((row, index) => (
          <S.BodyRow
            key={String(row.id)}
            striped={striped && index % 2 === 1}
            selected={isSelected(row)}
            onClick={() => onRowClick?.(row)}
          >
            {selectable && (
              <S.CheckboxCell>
                <input
                  type="checkbox"
                  checked={isSelected(row)}
                  onChange={() => toggleRowSelection(row)}
                />
              </S.CheckboxCell>
            )}
            {columns.map(col => (
              <S.BodyCell key={col.accessor} align={col.align}>
                {handleCellRender(row, col)}
              </S.BodyCell>
            ))}
          </S.BodyRow>
        ))}
      </S.BodyWrapper>

      <S.Footer>
        <S.Cell>총 {displayData.length}건</S.Cell>
        {selectable && (
          <S.Cell>
            선택 {selectedIds.size}건
          </S.Cell>
        )}
      </S.Footer>
    </S.GridContainer>
  );
}
`,
};

function generateLargeCodebase(seed: number): string {
  // seed 기반의 의사 난수 생성기 (LCG)
  let state = seed;
  const rand = () => {
    state = (state * 1664525 + 1013904223) & 0xffffffff;
    return (state >>> 0) / 0xffffffff;
  };

  const files = Object.values(CODEBASE_TEMPLATES);
  const chunks: string[] = [];

  // 코드베이스 헤더
  chunks.push(
    '// ========================================',
    '// Large Codebase Context (benchmark payload)',
    `\`// Generated: seed=\${seed}\``,
    '// ========================================',
    ''
  );

  // 각 템플릿을 그대로 포함 + 반복하여 충분히 크게 만듦
  const sections = [
    'types', 'service', 'controller', 'repository', 'middleware', 'hook', 'component'
  ];

  for (let iteration = 0; iteration < 3; iteration++) {
    chunks.push(\`/* --- Iteration \${iteration + 1} / 3 --- */\`);
    chunks.push('');

    for (const key of sections) {
      const template = CODEBASE_TEMPLATES[key];
      chunks.push(template.trim());
      chunks.push('');
      chunks.push('');
    }
  }

  // 추가 패딩: 더 많은 utility 함수 생성
  chunks.push('/* --- Utility Module --- */');
  chunks.push('');

  for (let i = 0; i < 20; i++) {
    chunks.push(\`\`);
    chunks.push(\`// utils/module_\${i + 1}.ts\`);
    chunks.push(\`export const MODULE_ID_\${i + 1} = 'module-\${String.fromCharCode(97 + (i % 26))}\${i}';\`);
    chunks.push('');
    chunks.push('export interface ModuleConfig {');
    for (let j = 0; j < 10; j++) {
      chunks.push(\`  field\${j + 1}: string | number | boolean | null;\`);
    }
    chunks.push('}');
    chunks.push('');
    chunks.push('export function processModuleData(input: ModuleConfig): ProcessedResult {');
    chunks.push('  const result = {');
    for (let j = 0; j < 10; j++) {
      chunks.push(\`    field\${j + 1}: input.field\${j + 1} != null ? String(input.field\${j + 1}) : 'default',\`);
    }
    chunks.push('  };');
    chunks.push('  return result as unknown as ProcessedResult;');
    chunks.push('}');
    chunks.push('');
    chunks.push('export function validateModuleConfig(config: Partial<ModuleConfig>): boolean {');
    chunks.push('  const requiredFields = [');
    for (let j = 0; j < 10; j++) {
      chunks.push(\`    'field\${j + 1}',\`);
    }
    chunks.push('  ];');
    chunks.push('  return requiredFields.every(f => config[f as keyof ModuleConfig] != null);');
    chunks.push('}');
    chunks.push('');
  }

  return chunks.join('\n');
}

const PREFIX_POOL = [
  "먼저 오늘의 날짜는 {date}입니다. ",
  "참고로 현재 사용자의 위치는 {city} 지역입니다. ",
  "이 요청의 고유 식별자는 #{id} 입니다. ",
  "다음 정보를 참고해주세요: 항목 수가 {n}개 있습니다. ",
  "현재 처리 중인 작업 순번은 {n} 번째입니다. ",
  "추가 맥락: 사용자 그룹 {g}에 속한 멤버의 요청입니다. ",
  "배경 정보: 지난 {n}일 동안 수집된 데이터를 바탕으로 합니다. ",
  "이 요청과 관련된 문서 ID는 DOC-{id} 입니다. ",
  "참고 데이터: 총 {n}명의 응답자가 참여한 설문 결과입니다. ",
  "현재 세션 정보: 세션 번호 SESS-{id}로 처리 중입니다. ",
];

const SUFFIX_POOL = [
  "결과를 설명할 때 반드시 #{id}라는 참조 번호를 포함해주세요.",
  "추가로 관련 분야 {topic}에 대한 짧은 의견도 함께 작성해주세요.",
  "마무리로 1부터 {n}까지의 숫자 총합도 함께 계산해서 알려주세요.",
  "가능하다면 {topic} 관점에서 어떻게 해석할지 짧게 덧붙여주세요.",
  "답변 마지막에 '참조: #{id}' 라는 문구를 꼭 포함시켜주세요.",
  "추가 요청: 위 내용을 {n}개의 키워드로 요약해서到最后에 붙여주세요.",
  "마지막으로 {city} 지역에서 이 주제를 적용할 수 있는 방법을 한 줄 써주세요.",
  "답변에 {topic}과 이 주제의 연관성을 한 문장 정도로 설명해주세요.",
  "별도로 #{id}번 요청에 대한 답변임을 마지막에 명시해주세요.",
  "위 내용과 관련하여 {n}개의 참고 문헌을 가상으로 구성해서 알려주세요.",
];

const TOPICS = [
  "양자컴퓨팅", "블록체인", "로보틱스", "생체공학", "나노기술",
  "에너지 저장", "우주개발", "디지털 트윈", "엣지컴퓨팅", "메타버스",
  "기후변화", "지속가능성", "순환경제", "탄소중립", "신재생에너지",
  "데이터프라이버시", "사이버보안", "클라우드네이티브", "자동화", "스마트시티",
];

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function varyPrompt(basePrompt, includeCodebase = false) {
  const id = Math.floor(Math.random() * 999999);
  const n = Math.floor(Math.random() * 900) + 100;
  const city = pickRandom(["서울", "부산", "인천", "대구", "광주", "대전", "울산", "제주", "수원", "익산", "춘천", "강릉"]);
  const topic = pickRandom(TOPICS);
  const date = `202${Math.floor(Math.random() * 6)}-${String(Math.floor(Math.random() * 12) + 1).padStart(2, "0")}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, "0")}`;
  const group = String.fromCharCode(65 + Math.floor(Math.random() * 26)) + Math.floor(Math.random() * 20);

  const prefix = pickRandom(PREFIX_POOL)
    .replace("{id}", id).replace("{n}", n).replace("{city}", city).replace("{g}", group).replace("{date}", date);
  const suffix = pickRandom(SUFFIX_POOL)
    .replace("{id}", id).replace("{n}", n).replace("{city}", city).replace("{topic}", topic);

  let codebaseContext = "";
  if (includeCodebase) {
    const seed = Math.floor(Math.random() * 999999);
    codebaseContext = `\n\n<!-- 참조 코드베이스 -->\n${generateLargeCodebase(seed)}\n`;
  }

  return `${codebaseContext}${prefix}본 질문: ${basePrompt} ${suffix}`;
}

// 현재 측정 상태
let measuring = false;
let measuringConfig = null;
let results = [];
let autoState = null;
let startTime = 0;

// SSE 스트림 리스너 관리
let streamListeners = [];

function broadcastSSE(event, data) {
  const msg = `event: ${event}\ndata: ${JSON.stringify(data, null, 0)}\n\n`;
  for (const ws of streamListeners) {
    ws.write(msg, "utf8");
  }
}

app.get("/api/stream", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.write(": connected\n\n", "utf8");
  streamListeners.push(res);

  req.on("close", () => {
    streamListeners = streamListeners.filter((s) => s !== res);
  });
});

app.post("/api/measure", async (req, res) => {
  if (measuring) {
    return res.json({ error: "측정이 이미 진행 중입니다" });
  }

  const {
    baseUrl,
    model,
    prompt,
    mode = "manual",
    concurrent = 1,
    maxTokens = 1024,
    temperature = 0.7,
    rounds = 1,
    apiKey,
    autoMaxConcurrent = 100,
    autoWarmup = 2,
    autoPerRound = 3,
    varyPrompt = true,
    includeCodebase = false,
  } = req.body;

  if (!baseUrl || !model || !prompt) {
    return res.status(400).json({ error: "필수 입력값이 누락되었습니다" });
  }

  measuring = true;
  results = [];
  startTime = Date.now();
  autoState = null;

  if (mode === "auto") {
    measuringConfig = {
      mode: "auto",
      model,
      baseUrl,
      autoMaxConcurrent,
      autoPerRound,
    };
    runAutoBenchmark(
      baseUrl,
      model,
      prompt,
      maxTokens,
      temperature,
      apiKey,
      autoMaxConcurrent,
      autoWarmup,
      autoPerRound,
      varyPrompt,
      includeCodebase
    ).finally(() => {
      measuring = false;
      broadcastSSE("bench-complete", { reason: "done" });
    });
  } else {
    measuringConfig = {
      mode: "manual",
      concurrent,
      rounds,
      model,
      baseUrl,
    };
    runManualBenchmark(
      baseUrl,
      model,
      prompt,
      concurrent,
      maxTokens,
      temperature,
      rounds,
      apiKey,
      varyPrompt,
      includeCodebase
    ).finally(() => {
      measuring = false;
      broadcastSSE("bench-complete", { reason: "done" });
    });
  }

  res.json({ ok: true, message: "측정이 시작되었습니다" });
});

app.post("/api/stop", (_req, res) => {
  measuring = false;
  broadcastSSE("bench-complete", { reason: "stopped" });
  res.json({ ok: true, message: "측정 중지됨" });
});

app.get("/api/status", (_req, res) => {
  if (measuring || results.length > 0) {
    const elapsed = (Date.now() - startTime) / 1000;
    const totalTokens = results.reduce((s, r) => s + r.outputTokens, 0);
    const avgTPS = elapsed > 0 ? totalTokens / elapsed : 0;

    const resp = {
      measuring,
      completed: results.length,
      totalTokens,
      elapsed: Math.round(elapsed * 100) / 100,
      avgTPS: Math.round(avgTPS * 100) / 100,
      config: measuringConfig,
    };

    if (autoState) {
      resp.autoState = autoState;
    }

    return res.json(resp);
  }
  res.json({
    measuring: false,
    completed: 0,
    totalTokens: 0,
    elapsed: 0,
    avgTPS: 0,
    config: null,
  });
});

app.get("/api/results", (_req, res) => {
  const elapsed = (Date.now() - startTime) / 1000;
  const totalTokens = results.reduce((s, r) => s + r.outputTokens, 0);
  const avgTPS = elapsed > 0 ? totalTokens / elapsed : 0;

  const summary = {
    totalRequests: results.length,
    totalTokens,
    elapsed: Math.round(elapsed * 100) / 100,
    avgTPS: Math.round(avgTPS * 100) / 100,
    avgLatency:
      results.length > 0
        ? Math.round(
            results.reduce((s, r) => s + r.latency, 0) / results.length * 100
          ) / 100
        : 0,
    minTPS:
      results.length > 0
        ? Math.round(Math.min(...results.map((r) => r.tps)) * 100) / 100
        : 0,
    maxTPS:
      results.length > 0
        ? Math.round(Math.max(...results.map((r) => r.tps)) * 100) / 100
        : 0,
  };

  if (autoState) {
    summary.autoProfile = autoState.profile;
    summary.optimalConcurrent = autoState.optimalConcurrent;
    summary.optimalTPS = autoState.optimalTPS;
  }

  res.json({ results, summary });
});

app.get("/api/reset", (_req, res) => {
  measuring = false;
  results = [];
  autoState = null;
  measuringConfig = null;
  broadcastSSE("bench-complete", { reason: "reset" });
  for (const s of streamListeners) {
    s.end();
  }
  streamListeners = [];
  res.json({ ok: true });
});

// ── Manual benchmark ──────────────────────────────────────────

async function runManualBenchmark(
  baseUrl,
  model,
  prompt,
  concurrent,
  maxTokens,
  temperature,
  rounds,
  apiKey,
  varyPrompt,
  includeCodebase
) {
  const totalRuns = concurrent * rounds;
  let completed = 0;

  for (let round = 0; round < rounds && measuring; round++) {
    const promises = [];
    for (let i = 0; i < concurrent; i++) {
      promises.push(
        runSingleRequest(
          { baseUrl, model, prompt, maxTokens, temperature, apiKey, varyPrompt, includeCodebase },
          round * concurrent + i
        )
      );
    }

    const roundResults = await Promise.allSettled(promises);
    for (const r of roundResults) {
      if (r.status === "fulfilled") {
        results.push(r.value);
        completed++;
        console.log(
          `[${completed}/${totalRuns}] TPS: ${r.value.tps.toFixed(2)}, Latency: ${r.value.latency.toFixed(2)}s, Tokens: ${r.value.outputTokens}`
        );
      } else {
        console.error(`요청 실패:`, r.reason?.message || r.reason);
      }
    }
  }
}

// ── Auto benchmark ────────────────────────────────────────────
// Concurrency를 단계별로 늘려가며 TPS를 측정하고,
// 전체 TPS(동시접속 수 × 단일 TPS)가 가장 높은 지점을 찾습니다.
//
// 단계: 1 → 2 → 4 → 8 → 16 → 32 → 64 → (max 제한)
// 각 단계에서 여러 번 측정하여 평균을 내고,
// 전 단계 대비 전체 TPS가 90% 이하로 떨어지면 종료.

async function runAutoBenchmark(
  baseUrl,
  model,
  prompt,
  maxTokens,
  temperature,
  apiKey,
  maxConcurrent,
  warmupRounds,
  perLevelRounds,
  varyPrompt,
  includeCodebase
) {
  const profile = []; // [{ concurrent, avgTPS, totalTPS, avgLatency, success, failed }]
  let peakTotalTPS = 0;
  let peakConcurrent = 1;

  autoState = {
    phase: "warmup",
    currentConcurrent: 1,
    peakTotalTPS: 0,
    peakConcurrent: 1,
    optimalConcurrent: null,
    optimalTPS: null,
    profile,
    done: false,
  };

  // Step 1: warmup — concurrency=1 로 warmup 횟수 측정 (모델 로딩 등 제외)
  console.log("[Auto] Warmup 시작");
  const warmupResults = await runConcurrencyLevel(
    { baseUrl, model, prompt, maxTokens, temperature, apiKey },
    1,
    warmupRounds,
    varyPrompt,
    includeCodebase
  );
  if (warmupResults.failed > 0 && warmupResults.success === 0) {
    console.error("[Auto] Warmup 실패 — 모델 연결을 확인하세요");
    autoState.done = true;
    results.push(...warmupResults.results);
    return;
  }
  results.push(...warmupResults.results);

  // Concurrency 단계: 1, 2, 4, 8, 16, 32, 64, ...
  const steps = [1];
  for (let c = 2; c <= maxConcurrent; c *= 2) steps.push(c);
  if (!steps.includes(maxConcurrent)) steps.push(maxConcurrent);
  steps.sort((a, b) => a - b);

  for (const conc of steps) {
    if (!measuring) break;

    autoState.phase = "testing";
    autoState.currentConcurrent = conc;
    console.log(`[Auto] 동시접속=${conc} 측정 중...`);

    const levelResults = await runConcurrencyLevel(
      { baseUrl, model, prompt, maxTokens, temperature, apiKey },
      conc,
      perLevelRounds,
      varyPrompt,
      includeCodebase
    );
    results.push(...levelResults.results);

    if (levelResults.success === 0) {
      console.error(`[Auto] 동시접속=${conc} 전체 실패`);
      profile.push({
        concurrent: conc,
        avgTPS: 0,
        totalTPS: 0,
        avgLatency: 0,
        success: 0,
        failed: levelResults.failed,
      });
      autoState.profile = profile;
      break;
    }

    // totalTPS = 동시접속 시 모든 요청의 TPS 합계
    const totalTPS =
      levelResults.results.reduce((s, r) => s + r.tps, 0);
    const avgTPS = totalTPS / levelResults.success;
    const avgLatency =
      levelResults.results.reduce((s, r) => s + r.latency, 0) /
      levelResults.success;

    profile.push({
      concurrent: conc,
      avgTPS: Math.round(avgTPS * 100) / 100,
      totalTPS: Math.round(totalTPS * 100) / 100,
      avgLatency: Math.round(avgLatency * 100) / 100,
      success: levelResults.success,
      failed: levelResults.failed,
    });

    autoState.profile = profile;

    if (totalTPS > peakTotalTPS) {
      peakTotalTPS = totalTPS;
      peakConcurrent = conc;
      autoState.peakTotalTPS = Math.round(peakTotalTPS * 100) / 100;
      autoState.peakConcurrent = peakConcurrent;
    }

    console.log(
      `[Auto] 동시접속=${conc} totalTPS=${totalTPS.toFixed(
        1
      )}, peakTotalTPS=${peakTotalTPS.toFixed(1)}`
    );

    // Degradation 감지: peak 대비 90% 미만이면 종료
    if (
      conc > peakConcurrent &&
      totalTPS < peakTotalTPS * 0.9
    ) {
      console.log(
        `[Auto] TPS 저하 감지 — 최적 동시접속=${peakConcurrent} (peak TPS=${peakTotalTPS.toFixed(1)})`
      );
      break;
    }
  }

  autoState.phase = "complete";
  autoState.done = true;
  autoState.optimalConcurrent = peakConcurrent;
  autoState.optimalTPS = Math.round(peakTotalTPS * 100) / 100;
  console.log(
    `[Auto] 완료 — 최적 동시접속=${peakConcurrent}, TPS=${peakTotalTPS.toFixed(1)}`
  );
}

async function runConcurrencyLevel(options, concurrent, rounds, varyPrompt, includeCodebase) {
  let success = 0;
  let failed = 0;
  const levelResults = [];

  for (let round = 0; round < rounds && measuring; round++) {
    const promises = [];
    for (let i = 0; i < concurrent; i++) {
      promises.push(
        runSingleRequest(
          { ...options, varyPrompt, includeCodebase },
          levelResults.length + failed + i
        )
      );
    }

    const settled = await Promise.allSettled(promises);
    for (const r of settled) {
      if (r.status === "fulfilled") {
        levelResults.push(r.value);
        success++;
      } else {
        failed++;
        console.error(`  요청 실패:`, r.reason?.message || r.reason);
      }
    }
  }

  return { results: levelResults, success, failed };
}

// ── Shared helpers ────────────────────────────────────────────

function createChatModel(options) {
  const config = {
    model: options.model,
    temperature: options.temperature,
    maxTokens: options.maxTokens,
    configuration: {
      baseURL: options.baseUrl,
    },
  };
  if (options.apiKey) {
    config.apiKey = options.apiKey;
  }
  return new ChatOpenAI(config);
}

async function runSingleRequest(options, index) {
  const t0 = Date.now();
  const chat = createChatModel(options);

  broadcastSSE("request-start", { id: index });

  try {
    const actualPrompt = options.varyPrompt
      ? varyPrompt(options.prompt, options.includeCodebase)
      : options.prompt;
    const stream = await chat.stream([new HumanMessage(actualPrompt)]);
    let fullText = "";
    let usageMetadata = null;

    for await (const chunk of stream) {
      // stream() yields AIMessageChunk objects
      // content can be string or array of content parts
      let text = "";
      const raw = chunk.content || chunk.message?.content || "";
      if (typeof raw === "string") text = raw;
      else if (Array.isArray(raw)) text = raw.map((p) => p.text || p.value || "").join("");

      if (text) {
        fullText += text;
        broadcastSSE("token", { id: index, text });
      }
      // usage_metadata uses snake_case keys (output_tokens, input_tokens, total_tokens)
      if (chunk.usage_metadata) {
        usageMetadata = chunk.usage_metadata;
      }
    }

    const t1 = Date.now();
    const latency = (t1 - t0) / 1000;
    const outputTokens = usageMetadata?.output_tokens || 0;
    const totalTokens = usageMetadata?.total_tokens || 0;
    const inputTokens = usageMetadata?.input_tokens || 0;
    const tps = latency > 0 ? outputTokens / latency : 0;

    broadcastSSE("request-end", {
      id: index,
      latency: Math.round(latency * 100) / 100,
      outputTokens,
      inputTokens,
      totalTokens,
      tps: Math.round(tps * 100) / 100,
    });

    return {
      index,
      latency: Math.round(latency * 100) / 100,
      outputTokens,
      inputTokens,
      totalTokens,
      tps: Math.round(tps * 100) / 100,
      timestamp: new Date().toISOString(),
    };
  } catch (err) {
    broadcastSSE("request-error", { id: index, error: err.message || String(err) });
    throw err;
  }
}

const PORT = process.env.PORT || 3100;
app.listen(PORT, () => {
  console.log(`LLM Bench 서버 시작: http://localhost:${PORT}`);
});
