/**
 * 设备检测工具函数
 */

/**
 * 检测是否为移动设备
 * 综合考虑 UA、触摸支持、屏幕尺寸
 */
export function isMobileDevice(): boolean {
  if (typeof navigator === "undefined" || typeof window === "undefined") {
    return false;
  }

  const ua = navigator.userAgent || navigator.vendor || (window as any).opera || "";
  
  // UA 检测
  const isMobileUA = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(
    ua.toLowerCase()
  );
  
  // 触摸支持检测
  const hasTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
  
  // 屏幕尺寸检测
  const isSmallScreen = window.innerWidth <= 768;
  
  // iPad as Mac 检测（Safari on iPad）
  const isIPadAsMac = navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
  
  return isMobileUA || isIPadAsMac || (hasTouch && isSmallScreen);
}

/**
 * 获取推荐的设备像素比
 * 移动端限制 DPI 以避免性能问题
 */
export function getRecommendedDPR(): number {
  const isMobile = isMobileDevice();
  const maxDpr = isMobile ? 1.5 : 3;
  return Math.min(window.devicePixelRatio || 1, maxDpr);
}

/**
 * 检测 WebGPU 支持
 */
export function isWebGPUSupported(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}
