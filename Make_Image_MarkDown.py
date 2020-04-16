

# ***        건들지 말기    ****
link = "https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/"
last_str = ".jpg?raw=true"
a = '_'
mark = "![img]("
b = ')'
c =  '/'

# ***         수정 필요     ****ㅠ
page_num = 38 # Git에 올라간 파일 중 가장 마지막 번호 그대로!
image_name = "lec-5" # _num.jpg를 제외한 이미지 이름 (pdf 이름 그대로 이다.)
link_date = "2020-04-17"  # Git repository 의 폴더 : Ex) "2020-04-12/"


# ***        건들지 말기    ****
for i in range(1, page_num+1):
    str_complete = link + link_date + c + image_name + a + str(i) + last_str
    full_str = mark + str_complete + b
    print(full_str)  


    