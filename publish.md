# NPM 发布流程

## 前置准备

1. 确保已安装 Node.js 和 npm
2. 注册 npm 账号：https://www.npmjs.com/signup

## 首次发布

### 1. 登录 npm

```bash
npm login
```

按提示输入用户名、密码和邮箱。

### 2. 检查包名是否可用

```bash
npm search @d5techs/3dgs-lib
```

### 3. 构建项目

```bash
yarn build:lib
```

### 4. 预览发布内容

```bash
npm pack --dry-run
```

确认只包含必要文件（dist 目录）。

### 5. 发布

```bash
npm publish --access public
```

## 版本更新发布

### 1. 更新版本号

```bash
# 补丁版本 1.0.0 -> 1.0.1
npm version patch

# 次版本 1.0.0 -> 1.1.0
npm version minor

# 主版本 1.0.0 -> 2.0.0
npm version major
```

### 2. 发布

```bash
npm publish
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `npm whoami` | 查看当前登录用户 |
| `npm view @d5techs/3dgs-lib` | 查看已发布包信息 |
| `npm deprecate @d5techs/3dgs-lib@1.0.0 "message"` | 废弃某版本 |
| `npm unpublish @d5techs/3dgs-lib@1.0.0` | 删除某版本（72小时内） |

## 注意事项

- 发布前会自动执行 `prepublishOnly` 脚本进行构建
- 只有 `files` 字段指定的 `dist` 目录会被发布
- 发布后无法修改，只能发布新版本
