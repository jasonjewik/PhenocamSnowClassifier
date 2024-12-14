$(document).ready(function() {
    $(window).keydown(function (event) {
	switch (event.which) {
	case 39:
	    window.location.href = $('#nextDay')[0].href;
	    break;
	case 37:
	    window.location.href = $('#prevDay')[0].href;
	    break;
	};
    });
});
