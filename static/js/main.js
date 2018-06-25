$(function(){
    var arr = new Array();

    $('#slidebars  #slidebar').slider({
        min: 0,
        max: 1,
        step: 0.001,
        value: 0,
        slide: function(e, ui) {
            $(this).prev('.nval').html('<span class="nval">' + ui.value + '</span>');
        },
        change: function(e, ui) {
            $(this).prev('.nval').html('<span class="nval">' + ui.value + '</span>');
            // console.log("name: " + Number($(this).attr('name')));
            arr[Number($(this).attr('name'))] = ui.value;
            // console.log("change: " + arr);

            $.ajax({
                url: '/chainer',
                type: 'post',
                data: {
                    range: arr
                },
            })
            .done(function(response) {
                var base = 'data:image/png;base64,' + response;
                $('.image').children('img').attr('src', base);
            });

        },
        create: function(e, ui) {
            $(this).prev('.nval').html('<span class="nval">' + $(this).slider('option', 'value') + '</span>');
            arr.push($(this).slider('option', 'value'));
            // console.log("create: " + arr);
        }
    });
});