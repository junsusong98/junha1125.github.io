---
layout: post
title: 【Git】Git-blog PRO VERSION check list

---

Git-blog 프로버전으로 업그레이드 하고 파악 했던 내용들을 정리해 놓는다. 

# Blog PRO-v check list & google analystics

**prerequisites** : Jekyll 설치, Ruby 설치.

**PS** : [제작자 사이트](https://hydejack.com/)



1. 설치 및 구동

   - 다운 받은 zip파일에서 `starter-kit-gh-pages` 폴더를 그대로 사용했다.
   - 폴더 내부의 내용을 모두 `gitID.github.io` git pull 한 폴더에 옮겨 놓는다.
   - 해당pwd에서 terminal 실행 후 빌드: $ bundle install
   - 한글 인식 오류 나면 cmd창에서, $ chcp 65001
   - $ bundle exec jekyll serve : local host로 연결되는 link에서 사이트 잘 생성되는지 확인

2. 구성 요소 정리

   - `_config.yml` 내용 수정하기.
     - 특히 카테고리 관련 사항 주의
   - `_data/authors.yml` 내용 수정
   - `assets`에 필요한 이미지 넣어두기
   - `about.md` 수정하기
   - `_featured_categories` 폴더 내부에 카테고리 관련 내용 넣어두기

3. 카테고리별 폴더 정리

   - 카테고리안에 들어갈 post를 담기 위해, 폴더 생성

     ```
        - _featured_categories
        - category1
           - _post
              - md file1
              - md file2
        - category2
           - _post
              - md file1
              - md file2
     ```

   - md file은 .md 형식의 파일이며, markdown 형식으로 파일 기제

4. 추가적인, 폴더 정리 및 페이지 정리

   - ```
     _config.yml
     ```

      에서

     - copyright 문장 수정
     - cookies_banner : false

   - ```
     root 폴더
     ```

      내부에, 쓸모없는 예제 파일,폴더들 한방에 몰아 넣어 두기. 아래의 파일을 모두 

     ```
     docs
     ```

      폴더에 넣어두었음.

     - [offline.md](http://offline.md)
     - [posts.md](http://posts.md)
     - [projects.md](http://projects.md)
     - [forms-by-example.md](http://forms-by-example.md)
     - [NOTICE.md](http://NOTICE.md)
     - [LICENSE.md](http://LICENSE.md)
     - [CHANGELOG.md](http://CHANGELOG.md)
     - example 폴더
     - licenses 폴더
     - 특히 docs폴더에는 나중에 참고하면 좋을 예체 md파일들이 있으니.. 좀더 새로운 형식의 post를 원한다면, 이곳의 파일에서 사용한 syntax를 그대로 가져와 사용하기.

   - index.md

      수정하기

     - 매우 중요.
     - 블로그의 HOME이라고도 할 수 있다.

5. [resume.md](http://resume.md)

   - root/resume.md 파일과 _data/resume.yml 파일을 수정해서 제작자와 같은 resume형식 제작 가능 (하지만 나는 아직 사용 안함)
   - 블로그 제작자의 [resume](https://hydejack.com/resume/)

6. Update

   - 제공된 라이센스 키를 가지고 업데이트 제공받을 수 있음.
   - [제작자 사이트](https://hydejack.com/)
   - [junha1125.github.io](http://junha1125.github.io)\docs\for-future.pdf 파일 참조

## **Google Analytic를 활용한 접속자 조회 방법**

1. google analystic 가입하기

2. 관리 -> 데이터 스트림 -> 웹 사이트 등록(스트림 추가)해두기

   ![https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175437258.png?raw=tru](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175437258.png?raw=tru)

3. 내가 만든 스트림 선택해서 아래의 창에서 몇가지 수정하기

   ![https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175535751.png?raw=tru](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175535751.png?raw=tru)

4. 먼저 측정 ID를 복사해서 _config.yml 파일에 적어 넣어두기

   ```
   google_analytics:      G-68888888JQ
   ```

5. _includes\body\analytics.html 에 사이트 태그 복사해서 붙여 넣기

   ```
   ....
       w.ga(function(tracker) {
           w.ga("set", "an....eIp", undefined);
           localStorage && localStorage.setItem("g.....id", tracker.get("clientId"));
         });
       });
   
       w.loadJSDeferred('<https://www.google-analytics.com/analytics.js>');
     }(window, document);</script>
     
     <!-- Global site tag (gtag.js) - Google Analytics -->
     <script async src="<https://www.googletagmanager.com/gtag/js?id=G-6>....VJQ"></script>
     <script>
       window.dataLayer = window.dataLayer || [];
       function gtag(){dataLayer.push(arguments);}
       gtag('js', new Date());
   
       gtag('config', 'G-68...JQ');
     </script>
   ```

6. 3번 사진 맨 아래 연결된 사이트 태그에, 내 측정 ID등록해 두기

7. 그럼 접속자 조회가 가능하다.

   ![https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175906590.png?raw=tru](https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175906590.png?raw=tru)

이 접속자 조회로, 하루하루 접속자가 몇명이고 접속자들이 가장 많이 보는 게시물이 무엇이고 등등을 알 수 있다. 이것을 통해서 블로그 작성의 새로운 동기 부여가 될 수 있기...











---

---

## <수정 이전 버전>

1. 설치 및 구동

   - 다운 받은 zip파일에서 starter-kit-gh-pages 를 사용했다. 
   - 폴더 내부의 내용을 모두 다른 곳으로 옮기고
   - $ bundle install
   - (cmd창에서. git-bash 말고.) chcp 65001
   - $ bundle exec jekyll serve

2. 구성 요소 정리

   - _config.yml 내용 수정
     - 특히 카테고리 관련 사항 주의
   - _data/authors.yml 내용 수정
   - assets에 필요한 이미지 넣어두기
   - about.md 수정하기
   - _featured_categories 폴더 내부에 카테고리 관련 내용 넣어두기

3. 카테고리 폴더 정리

   - 카테고리안에 들어갈 post를 담기 위해, 폴더 생성   
      ```sh
         - _featured_categories
         - category1
            - _post
               - md file1
               - md file2
         - category2
            - _post
               - md file1
               - md file2
      ```

   - md file은 .md 형식의 파일이며, markdown 형식으로 파일 기제

4. Home 화면 정리

   - _config.yml 에서 
     - copyright 수정
     - cookies_banner : false
     - *# legal:* 부분에 title과 url 추가하면 화면 맨 아래 하이퍼링크 추가 가능
   - docs폴터 내부에, 쓸모없는 예제 파일들 몰아 넣어 두기 (root에 있던 파일들)
     - offline.md
     - posts.md
     - projects.md
     - forms-by-example.md
     - NOTICE.md
     - LICENSE.md
     - CHANGELOG.md
     - example 폴더
     - licenses 폴더
     - 특히 docs폴더에는 나중에 참고하면 좋을 예체 md파일들이 있으니.. 좀더 새로운 형식의 post를 원한다면, 이곳의 파일에서 사용한 syntax를 그대로 가져와 사용하기.
   - index.md 수정하기
     - 매우 중요. 
     - 블로그의 HOME이라고도 할 수 있다. 

5. resume.md

   - root/resume.md 파일과 _data/resume.yml 파일을 수정해서 제작자와 같은 resume형식 제작 가능
   - 블로그 제작자의 [resume](https://hydejack.com/resume/)

6. Update

   - 제공된 라이센스 키를 가지고 업데이트 제공받을 수 있음.
   - [제작자 사이트](https://hydejack.com/)
   - junha1125.github.io\docs\for-future.pdf 파일 참조





# Google Analytic를 활용한 접속자 조회 방법

1. google analystic 가입하기

2. 관리 -> 데이터 스트림 -> 웹 사이트 등록(스트림 추가)해두기    
   <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175437258.png?raw=tru" alt="image-20210127175437258" style="zoom: 67%;" />

3. 내가 만든 스트림 선택해서 아래의 창에서 몇가지 수정하기   
   <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175535751.png?raw=tru" alt="image-20210127175535751" style="zoom:80%;" />

4. 먼저 측정 ID를 복사해서 _config.yml 파일에 적어 넣어두기   

   ```sh
   google_analytics:      G-68888888JQ
   ```

   

5. _includes\body\analytics.html 에 사이트 태그 복사해서 붙여 넣기   

   ```sh
   ....
   	w.ga(function(tracker) {
           w.ga("set", "an....eIp", undefined);
           localStorage && localStorage.setItem("g.....id", tracker.get("clientId"));
         });
       });
   
       w.loadJSDeferred('https://www.google-analytics.com/analytics.js');
     }(window, document);</script>
     
     <!-- Global site tag (gtag.js) - Google Analytics -->
     <script async src="https://www.googletagmanager.com/gtag/js?id=G-6....VJQ"></script>
     <script>
       window.dataLayer = window.dataLayer || [];
       function gtag(){dataLayer.push(arguments);}
       gtag('js', new Date());
   
       gtag('config', 'G-68...JQ');
     </script>
   
   ```

   

6. 3번 사진 맨 아래 연결된 사이트 태그에, 내 측정 ID등록해 두기

7. 그럼 접속자 조회가 가능하다.  
   <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/Typora/image-20210127175906590.png?raw=tru" alt="image-20210127175906590" style="zoom:80%;" />

   

이 접속자 조회로, **하루하루 접속자가 몇명이고 접속자들이 가장 많이 보는 게시물**이 무엇이고 등등을 알 수 있다. 이것을 통해서 **블로그 작성의 새로운 동기 부여**가 될 수 있기를 바란다. 



