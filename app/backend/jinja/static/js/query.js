function showImage(data){
    if (data["mask"] == 1){
        alert("Tumor Detected");
    }
    else alert("Can't detect tumor");
    const path = data["image"];
    const doc = $("#picture_2");
    doc.attr("src",`../../static/image/image_after/${path}`);
}

document.querySelector(".btn-transport").addEventListener("click", function (){
    $.get(
        "image/segment",
        {
            name_picture : document.querySelector(".picture-name").value,
        }
    ).done(data => {showImage(data);});
    });