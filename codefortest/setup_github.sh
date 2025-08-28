#!/bin/bash

# GitHub配置脚本
# 请修改下面的变量为你的实际信息

# ========== 请修改这些变量 ==========
GITHUB_USERNAME="jingchengsimon"
GITHUB_EMAIL="jingcheng.shi@rutgers.edu"
REPOSITORY_NAME="neuron_as_TCN"
# =====================================

echo "🚀 开始配置GitHub..."
echo "用户名: $GITHUB_USERNAME"
echo "邮箱: $GITHUB_EMAIL"
echo "仓库名: $REPOSITORY_NAME"
echo ""

# 1. 配置Git用户信息
echo "📝 配置Git用户信息..."
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

# 2. 配置凭证缓存
echo "🔐 配置凭证缓存..."
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=3600'

# 3. 检查配置
echo "✅ 检查配置..."
echo "Git用户名: $(git config --global user.name)"
echo "Git邮箱: $(git config --global user.email)"
echo ""

# 4. 添加远程仓库
echo "🌐 添加远程仓库..."
REPO_URL="https://github.com/$GITHUB_USERNAME/$REPOSITORY_NAME.git"
git remote add origin "$REPO_URL"

echo "远程仓库已添加: $REPO_URL"
echo ""

# 5. 显示下一步操作
echo "🎯 下一步操作："
echo "1. 在GitHub上创建名为 '$REPOSITORY_NAME' 的仓库"
echo "2. 运行: git push -u origin master"
echo "3. 输入你的GitHub用户名和Personal Access Token"
echo ""
echo "📖 详细说明："
echo "- 创建仓库：https://github.com/new"
echo "- 生成Token：https://github.com/settings/tokens"
echo "- Token权限选择：repo, workflow"
echo ""

echo "✨ 配置完成！" 