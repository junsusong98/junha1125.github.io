---
layout: post
title: 【Git】Git-blog PRO VERSION check list

---

Git-blog 프로버전으로 업그레이드 하고 파악 했던 내용들을 정리해 놓는다. 

# Git blog PRO VERSION check list

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





