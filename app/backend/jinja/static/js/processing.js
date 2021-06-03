
$(".upload_img").on("change",  ev =>{
    var f = ev.target.files[0];
    var fr = new FileReader();

    fr.onload = ev2 =>{
        var fileName = "../static/image/job_conver/" + f.name;
        var doc = $("#picture_1");
        doc.attr("src", fileName);
        doc.attr("name", f.name);

    };
    fr.readAsDataURL(f);
})
