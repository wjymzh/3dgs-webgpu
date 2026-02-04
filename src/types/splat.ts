/**
 * 3D Gaussian Splatting 相关类型定义
 */

/**
 * SH 模式枚举 - 球谐函数级别
 */
export enum SHMode {
  L0 = 0,  // 仅 DC 颜色（最快）
  L1 = 1,  // DC + L1 SH
  L2 = 2,  // DC + L1 + L2 SH
  L3 = 3,  // 完整 SH（最高质量）
}

/**
 * 渲染器能力描述
 */
export interface RendererCapabilities {
  /** 支持的最高 SH 模式 */
  maxSHMode: SHMode;
  /** 是否支持原始 SplatCPU 数据 */
  supportsRawData: boolean;
  /** 是否为移动端优化版本 */
  isMobileOptimized: boolean;
  /** 最大支持的 splat 数量（0 表示无限制） */
  maxSplatCount: number;
}
