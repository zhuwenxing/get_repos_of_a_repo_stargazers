from github import Github
import csv
import time
import random
# First create a Github instance:

# using username and password
# g = Github("user", "password")

# or using an access token
# g = Github(token)

# Github Enterprise with custom hostname
# g = Github(base_url="https://{hostname}/api/v3", login_or_token="access_token")

# Then play with your Github objects:

g= Github("user","password")# 此处的token是一个字符类型的变量，可以通过github的设置生成.

def check(repo):
    """Check
    检查找到的repo是否符合条件
    此处设置的条件是repo的编程语言设置为Python或者Matlab。
    也可以对repo.description进行关键字过滤。
    """
    if repo.language in ["Python", 'Matlab']:
        return True

users = []
with open("stargazers.csv", "r",encoding="utf-8") as csvFile:
    reader = csv.reader(csvFile)
    for i, row in enumerate(reader):
        if i % 2 == 1: # stargazers.csv中的输出是每行之间有空行
            continue
        # print(len(row))
        user = row[0]
        users.append(user)
users = users[1:]  # 不包括第一行 第一行是抬头

print(len(users))
wait_time = 5.0 # 设置等待时间，防止爬取时出现网络错误。
count = 0
for user in users:
    with open("Repositories.md", "a+", encoding="utf") as f:     
        
        try:
            f.write(f"## {user}\n")
            for repo in g.get_user(user).get_repos():
                if check(repo):
                    print(f"* [{repo.name}]({repo.html_url}):\n{repo.description}\n")
                    f.write(f"* [{repo.name}]({repo.html_url}):\n{repo.description}\n")
        except Exception as e:
            print(f"第{count+1}个用户发生错误")
            print(e)
    count = count + 1

    print(f'完成第{count}个用户的分析')
    print('Sleeping for %i seconds' % (wait_time , ))
    time.sleep(wait_time + random.uniform(0, 3))






