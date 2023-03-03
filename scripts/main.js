document.querySelector("h1").addEventListener("click", function () {
    alert("这是一个小彩蛋！");
  });
  
  let myImage = document.querySelector('img');

myImage.onclick = function() {
    let mySrc = myImage.getAttribute('src');
    if(mySrc === 'images/register-image.png') {
      myImage.setAttribute('src', 'images/logon-image.png');
    } else {
      myImage.setAttribute('src', 'images/register-image.png');
    }
    alert("哦！你切换了图片！");
}

let myButton = document.querySelector('button');
let myHeading = document.querySelector('h1');
function setUserName() {
    let myName = prompt('请输入你的名字。');
    if(!myName) {
      setUserName();
    } else {
      localStorage.setItem('name', myName);
      myHeading.textContent = '你好，' + myName;
    }
  }
  
  if(!localStorage.getItem('name')) {
    setUserName();
  } else {
    let storedName = localStorage.getItem('name');
    myHeading.textContent = '欢迎回来，' + storedName;
  }

  myButton.onclick = function() {
    setUserName();
 }
 
  