
$(".upload_img").on("change",  ev =>{
    const files = ev.target.files;
    const formData = new FormData();
    formData.append("myFile",files[0]);

    fetch("/fer", {
        method: "GET",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error =>{
        console.error(error);
    })
})

// function showImage(data){
//     const path = data["image"];
//     const doc = $("#picture-2")
//     doc.attr("src",`../../static/image/image_after/${path}`);
// }

// document.querySelector(".btn-transport").addEventListener("click", function (){
//     $.get(
//         "image/segment",
//         {
//             name_picture : document.querySelector(".picture-name").value,
//         }
//     ).done(data => {showImage(data);});
//     });