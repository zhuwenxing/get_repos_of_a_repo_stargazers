## Motivation
这个项目的主要是出于自身的需求！因为自己一直在更新一个专注整理DL4Wirelss领域的Paper with Code的repository。当有新的stargazer的时候，我习惯去这个用户的主页看看，可能可以发现一些有意思的repository。
更重要的一点是这些stargazers既然star了这个项目，所以是对DL2Wireless这个领域是有研究或者是有兴趣的，事实也是如此！所以关注[Paper-with-Code-of-Wireless-communication-Based-on-DL](https://github.com/IIT-Lab/Paper-with-Code-of-Wireless-communication-Based-on-DL)这个repo的用户很多都是与通信或深度学习专业相关的同学。通过查看这些同学主页的repositories,可以挖掘出一些潜在相关的Repository。
当然这样就需要自己每天手动去看看有没有新的star,如果有，还需要每个都手动点进首页查看，这很浪费时间！同时最主要问题的是这样只能是眼睛扫描一遍，不能记录下来，之后可能就没有印象了！所以希望能通过Python去自动爬取！

## How to
利用Python和Github提供的API,在通过网上类似的项目，完成了自己这样一个想法。
强相关的一个项目就是[get-profile-data-of-repo-stargazers-graphql](https://github.com/n0vad3v/get-profile-data-of-repo-stargazers-graphql)。通过这份代码生成一个stargazers.csv的文件。
之后就是利用[PyGithub](https://github.com/PyGithub/PyGithub)的库去获取stargazaers.csv中每个用户的repositories。
通过repo.language进行了一个过去，只记录编程语言设置为Python和Matlab的repository。
最后就是对获取到的repositories作了一个format，```[repo.name](repo.url):repo.description``` ，算是自动化生成一个.md文件。


## Documentation
```fetch_stargazers.py```:获取项目的star用户,得到```stargazers.csv```文件
```fetch_repos.py```：爬取所有star用户的repositories，得到```Repositories.md```文件

*How to run*

* 根据需求更改```fetch_stargazer.py```中的args_repo，args_token更改为自己Github生成的token
* ```fetch_repos.py```中的```g = Github()```修改为自己用户名和密码，或者另一个登录的Token。
* 记得修改输出的Markdown和csv文件名。

## Todo
说实话，这个项目（算得上吗？XD）自己写的代码不算多，主要就是了解一下PyGithub这个库的用法，还有就是如何生成md文件。
但是过程中花费的时间还是在找wheel，比如[get-profile-data-of-repo-stargazers-graphql](https://github.com/n0vad3v/get-profile-data-of-repo-stargazers-graphql)。同时还找到一些关于Github有意思的论文，例如：[HiGitClass](https://github.com/yuzhimanhua/HiGitClass):Keyword-Driven Hierarchical Classification of GitHub Repositories。

所以后面的工作就是希望能设计一个更好的过滤器和分类器，对这些爬取到的repositories进行过滤和分类。