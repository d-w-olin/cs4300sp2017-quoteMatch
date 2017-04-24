function getWiki(author) {
    var url = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro=&explaintext=&titles=';
    url += encodeURIComponent(author);

    $.getJSON(url, function(data) {
        var pageid = Object.keys(data.query.pages)[0];
        var text = data.query.pages[pageid].extract;

        $("#wiki-text").html(text + "<br><br>Learn more <a target='_blank' href='https://en.wikipedia.org/?curid="+pageid+"'>here</a>.");
    })
    .fail(function() {
        var searchurl = "https://en.wikipedia.org/wiki/Special:Search?search=" + author.split(" ").join("+") + "&go=Go";

        $("#wiki-text").html("No wikipedia entries on " + author + " found.<br><br>Try clicking <a target='_blank' href='"+searchurl+"'>this link</a> to see if you can learn more.");
    });
}

$(".quote").on('click', function(){
    if ($(".active").length > 0) {
        $(".active")[0].classList.remove("active");
    }
    this.classList.add("active");

    var author = $(".active .author")[0].innerHTML.substr(3);
    console.log("learning more about " + author);
    getWiki(author);
})
