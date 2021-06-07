function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
async function printKey(key){
    $('#print-box-Bob').append(`<div class="user title" >${key}</div>`)
    $('#print-box-Alice').append(`<div class="user title" >${key}</div>`)
}

async function printvalue(value, data){
    console.log(value);
    $(`#print-box-${value}`).append(`<div class ="user txt">${data[value]}</div>`);
}
async function printValue(data){
    for(var k in data){
        if (data[k] instanceof Object){
            printKey(k);
            await sleep(1000);
            printValue(data[k]);
            await sleep(1000);
        }
        else{
            console.log("Hello");
            printvalue(k, data);
        }
    }
}

function getElementByName(name){
    var a = $(`#input-${name}`).val();
    return a
}

$.get('/..', function(res){
    console.log("res:", res);
})
document.querySelector("#btn-compute").addEventListener("click", function (){
    $('.print-box').empty();
    $.get(
        "hellman/compute",
        {
            a: getElementByName('a'),
            b: getElementByName('b'),
            module: getElementByName('module'),
            basex: getElementByName('base-x'),
            basey: getElementByName('base-y'),
            aliceSecretKey: getElementByName('alice'),
            bobSecretKey: getElementByName('bob'),
        }
    ).done(data => {printValue(data);});
    });

