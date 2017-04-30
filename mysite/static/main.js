$(".quote-content").on("click", function(){
    var currentQuote = this;
    console.log(currentQuote);

    if (currentQuote.classList.contains('active')) {
        currentQuote.classList.remove('active');
        $("#wikipedia").slideToggle();
    } else {
        if ($(".active").length > 0) {
            $(".active")[0].classList.remove("active");
        }
        currentQuote.classList.add("active");

        var author = $(".active .author")[0].innerHTML.substr(3);
        console.log("learning more about " + author);
        $.ajax({
            url: "getWiki",
            data: {
                "author": author
            },
            dataType: "json",
            beforeSend: function() {
                $("html, body").animate({ scrollTop: 0 }, "slow");
                $("#title-quote").html("");
                $("#title-quote-author").html("");
                $("#wiki-text").html("");
                $("#wiki-img").html("");
                $("#wiki-img").removeClass("no-img");
                $("#wiki-authors").html("");
                $("#wiki-topics").html("");
                $("#wikipedia").css("display", "flex");
                $("#loading").show();
            },
            success: function(data) {
                $("#wikipedia").css("display", "none");
                $("#title-quote").html(currentQuote.firstElementChild.innerHTML);
                $("#title-quote-author").html("- " + currentQuote.lastElementChild.innerHTML.substr(3));
                if (data.extraction == "") {
                    $("#wiki-text").html("Information on " + author + " unavailable. Try checking <a target='_blank' href='" + data.pageurl + "'>this link</a> to learn more.");
                } else {
                    $("#wiki-text").html(data.extraction + "<span class='wiki-link'> Read more on <a href='" + data.pageurl + "' target='_blank'>Wikipedia</a></span>");
                    if (data.src == "") {
                        $("#wiki-img").addClass("no-img");
                    } else {
                        $("#wiki-img").html("<img src='" + data.src + "' alt='" + author + "'>");   
                    }
                }
                if (data.authors.length != 0) {
                    var similar_authors = "<span class='title'>Similar authors: </span>";
                    data.authors.forEach(function(d, i) {
                        if (i != data.authors.length - 1) {
                            similar_authors += d + " &middot; "
                        } else {
                            similar_authors += d
                        }
                    });
                    $("#wiki-authors").html(similar_authors);
                }
                if (data.topics.length != 0) {
                    var similar_topics = "<span class='title'>Similar topics: </span>";
                    data.topics.forEach(function(d, i) {
                        if (i != data.topics.length - 1) {
                            similar_topics += d + " &middot; "
                        } else {
                            similar_topics += d
                        }
                    });
                    $("#wiki-topics").html(similar_topics);
                }
            },
            error: function() {
                $("#wiki-text").html("Sorry, something went wrong. Try again later.")
            },
            complete: function() {
                $("#loading").hide();
                $("#wikipedia").slideToggle();
            }
        });  
    }
});

function showHiddenContent() {
    if ($(".text-button + .hidden-content")[0].classList.length == 1) {
        // about to retoggle to hidden
        $(".text-button")[0].innerHTML = "I want to improve my results!";
    } else {
        $(".text-button")[0].innerHTML = "I don't want to improve my results";
    }

    $(".hidden-content").each(function(i, d) {
        d.classList.toggle("hidden");
    });
}

// function rocchio() {
//     relevant = []
//     $("input[name='rocchio']").each(function(i, d) {
//         if (d.checked) {
//             relevant.push(d.value);
//         }
//     })

//     $.ajax({
//         url: "rocchio",
//         data: {
//             "relevant": relevant
//         },
//         dataType: "json",
//         success: function(data) {
//             console.log("'success'");
//         }
//     });
// }