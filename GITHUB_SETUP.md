# 🚀 GitHub配置完整指南

## 📋 前置要求

1. **GitHub账号**：确保你有一个GitHub账号
2. **Git已安装**：确保本地已安装Git
3. **项目已初始化**：当前项目已经完成Git初始化

## 🔧 配置步骤

### 步骤1：修改配置脚本

#### Linux/Mac用户：
编辑 `setup_github.sh` 文件，修改以下变量：
```bash
GITHUB_USERNAME="你的GitHub用户名"
GITHUB_EMAIL="你的GitHub邮箱"
REPOSITORY_NAME="neuron_as_TCN"
```

#### Windows用户：
编辑 `setup_github.bat` 文件，修改以下变量：
```batch
set GITHUB_USERNAME=你的GitHub用户名
set GITHUB_EMAIL=你的GitHub邮箱
set REPOSITORY_NAME=neuron_as_TCN
```

### 步骤2：运行配置脚本

#### Linux/Mac：
```bash
chmod +x setup_github.sh
./setup_github.sh
```

#### Windows：
双击运行 `setup_github.bat`

### 步骤3：在GitHub上创建仓库

1. 访问：https://github.com/new
2. 仓库名：`neuron_as_TCN`
3. 描述：`Neuron as Temporal Convolutional Network (TCN)`
4. 选择：Public 或 Private
5. 不要勾选 "Add a README file"（我们已经有了）
6. 点击 "Create repository"

### 步骤4：生成Personal Access Token

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 设置名称：`neuron_tcn_access`
4. 选择权限：
   - ✅ `repo` (完整的仓库访问权限)
   - ✅ `workflow` (GitHub Actions)
5. 点击 "Generate token"
6. **重要**：复制生成的token（只显示一次！）

### 步骤5：推送代码到GitHub

```bash
# 推送代码
git push -u origin master

# 系统会提示输入：
# Username: 你的GitHub用户名
# Password: 粘贴你的Personal Access Token
```

## 🔐 认证方式说明

### Personal Access Token（推荐）
- **优点**：简单易用，适合个人项目
- **缺点**：需要定期更新
- **适用场景**：个人开发、学习项目

### SSH密钥（更安全）
- **优点**：更安全，无需输入密码
- **缺点**：配置稍复杂
- **适用场景**：长期项目、团队协作

## 🚨 常见问题

### Q1: 推送时提示认证失败
**解决方案**：
- 检查用户名是否正确
- 确认Personal Access Token是否有效
- 重新生成Token

### Q2: 远程仓库已存在
**解决方案**：
```bash
# 删除现有远程仓库
git remote remove origin

# 重新添加
git remote add origin https://github.com/用户名/仓库名.git
```

### Q3: 分支名称问题
**解决方案**：
```bash
# 如果GitHub使用main分支
git branch -M main
git push -u origin main
```

## 📱 验证配置

配置完成后，运行以下命令验证：

```bash
# 检查远程仓库
git remote -v

# 检查用户配置
git config --global --list | grep user

# 检查凭证配置
git config --global --list | grep credential
```

## 🎯 后续操作

1. **日常推送**：`git push`
2. **拉取更新**：`git pull`
3. **查看状态**：`git status`
4. **查看历史**：`git log --oneline`

## 📞 需要帮助？

如果遇到问题，可以：
1. 检查GitHub的官方文档
2. 查看错误信息的具体内容
3. 确认网络连接和防火墙设置

---

**祝你的项目成功上传到GitHub！** 🎉 