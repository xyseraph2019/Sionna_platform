# Git 版本管理建议

本文件用于说明如何为 `D:\platform` 项目建立 Git 版本管理。
项目当前还没有 `.git` 仓库，我已创建好 `.gitignore`，可以直接初始化。

## 1. 初始化仓库

在项目根目录执行：

```powershell
cd D:\platform
git init
git add .
git status
```

检查 `git status` 中应该**不包含** `out/`、`*.whl`、`__pycache__/` 等生成物/大文件。

## 2. 首次提交

```powershell
git add .
git commit -m "chore: initial commit for 5G link-level simulation platform"
```

如果还没有配置用户信息，先执行：

```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## 3. 建议的分支管理

```text
main        # 稳定可运行版本
dev         # 日常开发集成分支
feature/*   # 新功能/新算法分支，例如 feature/nn-precoder
fix/*       # 修复分支，例如 fix/ls-estimation-floor
```

常用流程：

```powershell
git checkout -b feature/xxx
# ... 修改代码 ...
git add .
git commit -m "feat: xxx"
git checkout dev
git merge feature/xxx
```

## 4. 提交信息建议

- `feat:` 新功能、新算法、新配置
- `fix:` 修复 bug
- `refactor:` 重构，不改行为
- `perf:` 性能优化
- `test:` 增加/修改测试
- `docs:` 文档
- `chore:` 构建、依赖、仓库维护

## 5. 发布/里程碑

在完成一轮实验或稳定版本后打 tag：

```powershell
git tag -a v0.1.0 -m "5G link-level simulation platform v0.1.0"
git push origin main --tags
```

## 6. 远程仓库

如果已有远程仓库：

```powershell
git remote add origin <remote-url>
git branch -M main
git push -u origin main
```

如果没有远程仓库，也可以先在本地用 Git 管理，之后需要时再关联远程。

## 7. 已加入 `.gitignore` 的内容

- Python 缓存：`__pycache__/`, `*.py[cod]`
- 虚拟环境：`.venv/`, `venv/`, `env/`
- IDE：`.idea/`, `.vscode/`
- 运行输出：`out/`
- 大文件：`*.whl`
- 日志：`*.log`, `*.err`

如果你希望把某些 `out/` 下的典型结果（例如 `out/results/summary.md`）也纳入版本管理，可以在 `.gitignore` 中改成只忽略部分子目录，例如：

```gitignore
out/results/scenarios/
out/results/figures/
out/dmimo/
```
