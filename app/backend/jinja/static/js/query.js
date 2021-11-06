function showImage(data){
    const doc = $("#picture_2");
    const path = data["image"];
    if (data["mask"] == 1){
        alert("Tumor Detected");
        doc.attr("src",`../../static/image/image_after/${path}`);
    }
    else {
        alert("Can't detect tumor");
        doc.attr("src", `../../static/image/image_before/${path}`);
    }
}

document.querySelector(".btn-transport").addEventListener("click", function (){
    $.get(
        "image/segment",
        {
            name_picture : document.querySelector(".picture-name").value,
        }
    ).done(data => {showImage(data);});
    });