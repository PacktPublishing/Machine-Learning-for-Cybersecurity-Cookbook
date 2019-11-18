// VARS
var socket;

var text = ["LOADING A-Mei-Zing THINGS",
            "WORKING MIRACLES",
            "RESURRECTING DEAD MEMES",
            "PREPARING FINAL FORM",
            "DISPATCHING CARRIER PIGEONS",
            "IS THIS THING ON..?",
            "CALLING OUT YOUR LIES",
            "WE'RE TESTING YOUR PATIENCE",
            "WHY DON'T YOU ORDER A SANDWICH",
            "THE BITS ARE FLOWING SLOWLY TODAY",
            "WAITING FOR APPROVAL FROM PROF. WEI",
            "WARMING UP THE PROCESSORS",
            "WORKING... WELL YOU KNOW...",
            "RECALCULATING PI...",
            "COMMENCING INFINITE LOOP",
            "REWINDING THE DVD",
            "BUY MORE RAM...",
            "GO READ A BOOK, I'M NEARLY FINISHED",
            "HACKING PROF. WEI'S SERVERS FOR MORE DATA..."
        ];

var loadingMessagesInterval;
// FUNCTIONS
function beginAnalysis()
{
    //Step one - begin ajax request to upload the file to server

    // replace upload form with progressbar
    $('#uploadFormDiv').css('display', 'none');
    $("#progressBarDiv").css('display', 'block');
    let baseline_start = $('#base_start').val()
    let baseline_end = $('#base_end').val()
    let critical_start = $('#crit_start').val()
    let critical_stop = $('#crit_end').val()

    upload_video_file();
}

function upload_video_file()
{
    $('.determinate').css('width', '0%');
    $('#uploadValue').text('0');

    $("#videoForm").ajaxSubmit({
        url: '/',
        type: 'post',
        xhr: function () {
          var xhr = new window.XMLHttpRequest();
          xhr.upload.addEventListener('progress', function (evt) {
              if (evt.lengthComputable) {
                  var percentComplete = evt.loaded / evt.total;
                  percentComplete = parseInt(percentComplete * 100);
                  $('#uploadValue').text(percentComplete + '%');
                  $('.determinate').css('width', percentComplete + '%');
              }
          }, false);
          return xhr;
        },
        success: function (data) {
            console.log('Successfully uploaded');

            // Setup funny loading messages and show loading view
            $(".dim_overlay").css('display', 'block');
            $(".progress").css('display', 'block');
            $(".dim_loading_overlay").css('display', 'block');

            loadingMessagesInterval = setInterval(() => {
                let index = Math.floor(Math.random() * text.length);
                $(".dim_loading_overlay").text(text[index])
            }, 2000);

            socket = io.connect('http://' + document.domain + ':' + location.port)

            // Setup event listeners
            socket.on('connect', () => {
                // we emit a connected message to let the client know that we are connected
                console.log("Connected to Server")
            })


            // Setup listeners in affectiva.js
            setupSockets();
        }
    })
}

// ON PAGE LOAD
$( () => {
    $(document).on('drop dragover', (e) => {
       e.preventDefault();
     });

     $('input.timepicker').timepicker({
        timeFormat: 'HH:mm:ss',
        defaultTime: '00:00:00',
        dynamic: false,
        dropdown: false,
        scrollbar: false
    });

});