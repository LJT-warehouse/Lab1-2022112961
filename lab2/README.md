## 项目要求 初始化

项目要求使用**虚拟环境**，若不会可参考**附录 1**

激活当前虚拟环境：

```
 ...>Scripts> activate.bat
```



编程要求：函数有中文**注释**



项目要求**使用 git**，放在 GitHub 上，且**每次**开始写代码时记得**从仓库拉取最新代码**

GitHub 地址：

```shell
 # https://github.com/wwyyx1213/teacher-information-manage-system.git
 # 将远程仓库的 URL 添加到本地仓库的配置中，使其与本地仓库关联
 git remote add origin https://github.com/wwyyx1213/teacher-information-manage-system.git
```

代码开始编写时需要**pull**，

```shell
git pull

# 使用 --no-commit 选项可以拉取更改但不自动合并，允许用户检查更改后再手动合并：
```



结束代码编写后需要检测是否有明显 bug，若无则**push**到 GitHub。

```shell
git add .	
# 或者  git add 文件名

git commit -m "提交提示信息"

# 将本地 main 分支的代码推送到远程仓库
git push -u origin main
```





数据库要求使用**MySQL**，ip地址：

```
 10.68.69.19
```

```
用户名：admin
登录密码：456666
'PORT': '3306'
```

**注意**：IP 不是 localhost









## 项目结束

白盒测试、黑盒测试

当项目开发完后，建议使用 **pip freeze>requirements.txt** 命令将项目的库依赖导出









# 附录

## 1. 安装、使用 virtualenv 虚拟环境

> 虚拟环境：[PyCharm Django Python 开发环境配置 详细教程 - FreeK0x00 - 博客园 (cnblogs.com)](https://www.cnblogs.com/wuhongbin/p/14318656.html)

（1）安装 virtualenv 虚拟环境：

```
 pip install virtualenv
```

**使用虚拟环境**：

（2）创建虚拟环境目录 testvir

```
 ...> virtualenv [testvir]
```

（3）进入虚拟环境目录

```
 ...> cd testvir
```

（4）查看目录

```
 ...> dir
```

（5）进入 Scripts 工作目录

```
 ...>testvir> cd Scripts
```

（6）激活当前虚拟环境

```
 ...>Scripts> activate.bat  或者  cd Scripts
```

（7）查看当前环境

```
 (testvir) ... >Scripts> pip list
```

（8）退出虚拟环境

```
 (testvir) ... >Scripts> deactivate.bat
```



## 2. 在 Django 中使用 MySQL 作为数据库

需要以下步骤：

### 安装 MySQL 驱动

安装 PyMySQL，这是 Python 连接 MySQL 的驱动：

```bash
pip install pymysql
```

### 配置 Django 项目

修改 Django 项目的 `settings.py` 文件，配置数据库连接信息：

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',  # 指定了要使用的数据库后端
        'NAME': 'teacher_info_manage',
        'USER': 'admin',
        'PASSWORD': '456666',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

### 连接测试

确保 Django 能够连接到 MySQL 数据库。可以通过以下命令运行 Django 的检查工具：

```bash
python manage.py check
```

如果没有错误信息，说明连接配置正确。



### 示例

以下是一个完整的示例：

#### 安装 PyMySQL

```bash
pip install pymysql
```

#### 配置 `settings.py`

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'mydatabase',
        'USER': 'root',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

#### 在 Django 中使用

在 Django 项目中，可以使用 Django 的 ORM 来操作数据库。例如：

```python
from django.db import models

class MyModel(models.Model):
    field1 = models.CharField(max_length=100)
    field2 = models.IntegerField()
```

运行 Django 的迁移命令来创建数据库表：

```bash
python manage.py makemigrations
python manage.py migrate
```

### 常见问题

- **连接失败**：检查 MySQL 服务是否运行，用户名、密码、数据库名是否正确。
- **权限不足**：确保 MySQL 用户有权限访问指定数据库。
- **驱动问题**：确保 pyMySQL 已正确安装。
- **防火墙限制**：如果 MySQL 服务器在远程主机上，确保防火墙允许连接到 MySQL 的端口（默认 3306）。





### 其他