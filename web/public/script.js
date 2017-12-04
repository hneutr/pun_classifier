$(document).ready(function () {

    var timeouts = [];
    var baseUrl = "http://ec2-54-159-184-109.compute-1.amazonaws.com:8081/";

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
        var pun = $(".pun-input").val().trim();

        if (pun == "") {
            alert("Enter text");
        }
        else {
            transitionPage();

            $(".loader").removeClass("hidden");
            $(".pun-info-container").removeClass("hidden");

            //Add the input from the input box to the header to describe the pun that is being searched
            $(".pun-search").text('"' + pun + '"');

            $.ajax({
                type: "POST",
                url: baseUrl + "detection",
                data: pun
            }).done(function (data) {
                $(".pun-detection-text").removeClass("hidden");

                var types = ["baseline", "features"];
                types.forEach(function (t) {
                    var isPun = data[t].pun * 100;
                    var elementName = "." + t + "-detection";
                    if (isPun > 50) {
                        $(elementName).append("" +
                            "<p><img class=\"success\" src=\"public/success.png\"/>\n" +
                            "There is a " + (data[t].pun * 100) + "% probability that this is in fact a pun!" +
                            " Probability of not being a pun is " + (data[t]['non-pun'] * 100) + "%.</p>")
                    }
                    else {
                        $(elementName).append("" +
                            "<p><img class=\"failure\" src=\"public/failure.png\"/>\n" +
                            "This is most likely not a pun. The probability of not being a pun is " + (data[t]['non-pun'] * 100) + "%. " +
                            " The probability that this is a pun is " + (data[t].pun * 100) + "%.</p>")
                    }
                });


                $(".pun-detection-spinner").addClass("hidden");
            });


            //Fake loading - will connect with actual call later
            timeouts.push(setTimeout(function () {
                $(".pun-type-text").removeClass("hidden");
                $(".pun-type-spinner").addClass("hidden");
            }, 6000));
            timeouts.push(setTimeout(function () {
                $(".pun-location-text").removeClass("hidden");
                $(".pun-location-spinner").addClass("hidden");
            }, 4000));
        }


    });

    $('button.reset').click(function (e) {
        transitionPage();

        timeouts.forEach(function (timeout) {
            clearTimeout(timeout);
        });


        $(".pun-detection-text").addClass("hidden");
        $(".baseline-detection").empty();
        $(".features-detection").empty();
        $(".pun-detection-spinner").removeClass("hidden");

        $(".pun-type-text").addClass("hidden");
        $(".pun-type-spinner").removeClass("hidden");

        $(".pun-location-text").addClass("hidden");
        $(".pun-location-spinner").removeClass("hidden");
    });

});