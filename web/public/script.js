$(document).ready(function () {

    var timeouts = [];
    var baseUrl = "http://ec2-54-159-184-109.compute-1.amazonaws.com:8081/";
    //var baseUrl = "http://localhost:8081/";

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
        var pun = $(".pun-input").val();

        if (pun.trim() == "") {
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
                    var isPun = (1 - data[t]['non-pun']) * 100;
                    var elementName = "." + t + "-detection";
                    if (isPun > 50) {
                        $(elementName).append("" +
                            "<p><img class=\"success\" src=\"public/success.png\"/>\n" +
                            "There is a " + (isPun) + "% probability that this is in fact a pun!" +
                            " Probability of not being a pun is " + (data[t]['non-pun'] * 100) + "%.</p>")
                    }
                    else {
                        $(elementName).append("" +
                            "<p><img class=\"failure\" src=\"public/failure.png\"/>\n" +
                            "This is most likely not a pun. The probability of not being a pun is " + (data[t]['non-pun'] * 100) + "%. " +
                            " The probability that this is a pun is " + (isPun) + "%.</p>")
                    }
                });

                $(".pun-type-text").removeClass("hidden");

                var types = ["baseline", "features"];
                types.forEach(function (t) {
                    var isNotPun = (data[t]['non-pun']) * 100;
                    //var isHomographic = (data[t]['homographic'] / (data[t]['homographic'] + data[t]['heterographic'])) * 100;
                    var isHomographic = ((data[t]['pun1'])/(data[t]['pun1'] + data[t]['pun2'])) * 100;
                    var elementName = "." + t + "-type_of_pun";
                    if(isNotPun > 50){
                      $(elementName).append("" +
                            "<p>This is most likely not a pun at all.</p>")
                    }
                    else{
                      if (isHomographic > 50) {
                        $(elementName).append("" +
                              "<p>This is most likely to be a Homographic pun. There was a " + (isHomographic) + "% chance that this was homographic, and there was a " + ((100 - isHomographic)) + "% chance that this was heterographic.</p>")
                      }
                      else {
                        $(elementName).append("" +
                              "<p>This is most likely to be a Heterographic pun. There was a " + (100 - isHomographic) + "% chance that this was heterographic, and there was a " + (isHomographic) + "% chance that this was homographic.</p>")
                      }
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
        $(".pun-detection-inner-text").remove();
        $(".pun-detection-spinner").removeClass("hidden");

        $(".pun-type-text").addClass("hidden");
        $(".pun-type-spinner").removeClass("hidden");

        $(".pun-location-text").addClass("hidden");
        $(".pun-location-spinner").removeClass("hidden");
    });

});
