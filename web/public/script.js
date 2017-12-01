$(document).ready(function () {

    var timeouts = [];

    function transitionPage() {
        var width = $(window).width();

        // Hide to left / show from left
        $("#pageone").toggle("slide", {direction: "left"}, width);

        // Show from right / hide to right
        $("#pagetwo").toggle("slide", {direction: "right"}, width);
    }


    if (window.getComputedStyle(document.body).mixBlendMode !== undefined) {
        $(".magnify").removeClass("hidden");
    }

    //Fixes weird css bug with skeleton
    $(".column").css('float', 'none');


    $('button.go').click(function (e) {

        transitionPage();

        $(".loader").removeClass("hidden");
        $(".pun-info-container").removeClass("hidden");

        //Add the input from the input box to the header to describe the pun that is being searched
        var pun = $(".pun-input").val();
        $(".pun-search").text('"' + pun + '"');


        //Fake loading - will connect with actual call later
        timeouts.push(setTimeout(function () {
            $(".pun-detection-text").removeClass("hidden");
            $(".pun-detection-spinner").addClass("hidden");
        }, 5000));
        timeouts.push(setTimeout(function () {
            $(".pun-type-text").removeClass("hidden");
            $(".pun-type-spinner").addClass("hidden");
        }, 6000));
        timeouts.push(setTimeout(function () {
            $(".pun-location-text").removeClass("hidden");
            $(".pun-location-spinner").addClass("hidden");
        }, 4000));
    });

    $('button.reset').click(function (e) {
        transitionPage();

        timeouts.forEach(function (timeout) {
            clearTimeout(timeout);
        });


        $(".pun-detection-text").addClass("hidden");
        $(".pun-detection-spinner").removeClass("hidden");

        $(".pun-type-text").addClass("hidden");
        $(".pun-type-spinner").removeClass("hidden");

        $(".pun-location-text").addClass("hidden");
        $(".pun-location-spinner").removeClass("hidden");
    });

});