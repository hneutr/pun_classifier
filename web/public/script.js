$(document).ready(function () {

    var timeouts = [];
    var baseUrl = "http://ec2-54-159-184-109.compute-1.amazonaws.com:8081/";

    Array.max = function( array ){
        return Math.max.apply( Math, array );
    };

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

                var types = ["baseline", "rnn", "features", "voting"];
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

            $.ajax({
                type: "POST",
                url: baseUrl + "type",
                data: pun
            }).done(function (data) {
                $(".pun-type-text").removeClass("hidden");

                var types = ["baseline", "features"];
                types.forEach(function (t) {
                    var nonPun = data[t]['non-pun'] * 100;
                    var homographic = data[t].homographic * 100;
                    var heterographic = data[t].heterographic * 100;
                    var elementName = "." + t + "-type";

                     $(elementName).append(
                         "<p>There is a " + nonPun + "% probability that this is not a pun at all.<br>" +
                         "There is a " + homographic + "% probability that this is a homographic pun.<br>" +
                         "There is a " + heterographic + "% probability that this is a heterographic pun.<br></p>")
                });

                $(".pun-type-spinner").addClass("hidden");
            });

            $.ajax({
                type: "POST",
                url: baseUrl + "location",
                data: pun
            }).done(function (data) {
                $(".pun-location-text").removeClass("hidden");

                var types = ["rnn", "sliding"];
                types.forEach(function (t) {

                    var elementName = "." + t + "-location";
                    var mostProbableElementName = elementName + "-most-probable";

                    var i = 0;
                    var total = 0;
                    var punArray = pun.replace(/[^\w\s]|_/g, "").replace(/\s+/g, " ").split(" ");

                    data[t] = data[t].map(function(ele) {
                        return parseFloat(ele);
                    });

                    var maxX = Array.max(data[t]);
                    var index = data[t].indexOf(maxX);
                    $(mostProbableElementName).append("<p> Most Probable Pun Word: " + punArray[index] + "</p>");

                    data[t].forEach(function (prediction) {
                       total  += parseFloat(prediction)
                    });

                    data[t].forEach(function (prediction) {
                        var color = "hsla(190012, 80%, 20%, percentage)".replace('percentage', parseFloat(prediction)/total);
                        $(elementName).append("<p>" + punArray[i] + "&nbsp;</p>");
                        $(elementName).children().eq(i).css({'font-weight': "bold", 'color': color});
                        i++;
                    });

                });

                $(".pun-location-spinner").addClass("hidden");
            });
        }


    });

    $('button.reset').click(function (e) {
        transitionPage();

        timeouts.forEach(function (timeout) {
            clearTimeout(timeout);
        });


        $(".pun-detection-text").addClass("hidden");
        $(".baseline-detection").empty();
        $(".rnn-detection").empty();
        $(".features-detection").empty();
        $(".voting-detection").empty();
        $(".pun-detection-spinner").removeClass("hidden");

        $(".pun-type-text").addClass("hidden");
        $(".baseline-type").empty();
        $(".features-type").empty();
        $(".pun-type-spinner").removeClass("hidden");

        $(".pun-location-text").addClass("hidden");
        $(".rnn-location").empty();
        $(".sliding-location").empty();
        $(".rnn-location-most-probable").empty();
        $(".sliding-location-most-probable").empty();
        $(".pun-location-spinner").removeClass("hidden");
    });

});