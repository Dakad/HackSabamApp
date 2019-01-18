$(function(){

    $(".clickable-row").on("click", function(){
        window.location = "/details?artiste="+$(this).attr("artiste") + "&date=" + $(this).attr("date") + "&place=" + $(this).attr("place");
    });

});
