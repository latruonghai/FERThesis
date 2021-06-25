
$(".upload_img").on("change",  ev =>{
    var f = ev.target.files[0];
    var fr = new FileReader();

    fr.onload = ev2 =>{
        var fileName = "../../static/image/image_before/" + f.name;
        var doc = $("#picture_1");

        var texts = $(".picture-name")
        texts.attr("value", f.name);
        doc.attr("src", fileName);
        doc.attr("name", f.name);

    };
    fr.readAsDataURL(f);
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