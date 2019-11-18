// socketIO

var faceMode = affdex.FaceDetectorMode.LARGE_FACES;

// Construct a FrameDetector and specify the image width / height and face detector mode.
var detector = new affdex.FrameDetector(faceMode);

// Affectiva timer timestamp
var startTimestamp;

// Canvas properties
var canvas_width;
var canvas_height;

var timeBetweenDrawings = 200;  //20ms

// Value is incremented when Facial data or Voice data is written
// When value == 2, automatically go to analysis page
var goToAnalysis = 0;

var setupSockets = () => {

  socket.on('canvas_width_height', (data) => {
    console.log('Server ready to receive frames');
    console.log('width, height: ', data);

    canvas_width = parseInt(data['width']);
    canvas_height = parseInt(data['height']);

    // Adjust canvas size to match video
    //document.getElementById("video_canvas").width = canvas_width
    //document.getElementById("video_canvas").height = canvas_height
    document.getElementById("dummy_canvas").width = canvas_width;
    document.getElementById("dummy_canvas").height = canvas_height;

    // Adjust physical canvas css width to 640 * 480
    //$("#video_canvas").css('width', 640)
    //$("#video_canvas").css('height', 480)

    socket.emit('ready_receive', {data: 'Ready Receive'});
  });

  socket.on('next_frame', (data) => {
    let frame_number = data[0];
    let base64_image = 'data:image/jpg;base64,' + data[1];

    // Get the dummy canvas that has been hidden on home page
    const canvas = document.getElementById("dummy_canvas");
    //const canvas2 = document.getElementById("video_canvas");
    const context = canvas.getContext('2d');
    //const context2 = canvas2.getContext('2d');

    var my_image = new Image();

    my_image.onload = () => {

      // bit base64 image unto both canvas and dummy canvas
      context.drawImage(my_image, 0, 0);
      // context2.drawImage(my_image, 0, 0)
      var imageData = context.getImageData(0, 0, canvas_width, canvas_height);

      // Affectiva requires timestamp, create initial before
      // analyzing entire video
      if (startTimestamp === undefined) 
      {
        startTimestamp = (new Date()).getTime() / 1000;
      }

      var now = (new Date()).getTime() / 1000;
      var deltaTime = now - startTimestamp;
      detector.process(imageData, deltaTime);
    };

    my_image.src = base64_image;

    console.log('Received Frame: ' + frame_number) 
  });

  socket.on('no_more_frames', () => {
    console.log('Complete');

      //Remove Dim Screen and Loading Messages
    $(".dim_overlay").css('display', 'none');
    $(".progress").css('display', 'none');
    $(".dim_loading_overlay").css('display', 'none');
    clearInterval(loadingMessagesInterval);
    stopEmotionDetection()
  });

  socket.on('data_complete', (message) => {

      if (message === 'Facial_Complete')
      {
         goToAnalysis++;

      } else if (message === 'Audio_Complete') {

          goToAnalysis++;
      }

      if (goToAnalysis === 2) {
          goToAnalysis = 0;
          socket.emit('client_request_disconnect', ' ');
          window.location.href = Flask.url_for("analysis");
      }
  });

};

function stopEmotionDetection()
{
  console.log("started stopping detector");
  if (detector && detector.isRunning) 
  {
    detector.removeEventListener();
    detector.stop();
    console.log("stopped detector")
  }
}


detector.addEventListener("onInitializeSuccess", () => {
  console.log("Affectiva loaded successfully");

  $(".overlay").html("");
  $(".file-field .btn").removeClass("disabled");
  //$(".results .btn-large").removeClass("disabled");
  $(".file-path").prop('disabled', false);
  $(".file-path").change( () => {

    if( $(".file-path").val().length > 0)
    {
      // Load Video locally using uploaded file
      var URL = window.URL || window.webkitURL
      var file = document.getElementById('fileItem').files[0]
      var fileURL = URL.createObjectURL(file)

      const video_player = document.getElementById("video")
      var source = document.createElement('source');
      source.setAttribute('src', fileURL);
      video_player.appendChild(source);
      video_player.load();
      $("#video").css('visibility', "visible");
      $(".results .btn-large").removeClass("disabled");

    }
  });

});

detector.addEventListener("onInitializeFailure", () => {
  console.log("Affectiva failed to load");
});

/* 
  onImageResults success is called when a frame is processed successfully and receives 3 parameters:
  - Faces: Dictionary of faces in the frame keyed by the face id.
           For each face id, the values of detected emotions, expressions, appearane metrics 
           and coordinates of the feature points
  - image: An imageData object containing the pixel values for the processed frame.
  - timestamp: The timestamp of the captured image in seconds.
*/
detector.addEventListener("onImageResultsSuccess", (faces, image, timestamp) => {
  console.log("Success processing image")

  if (faces.length > 0) 
  {
    let appearance = faces[0].appearance
    let emotions = faces[0].emotions  //key value pair dictionary of various 
    let expressions = faces[0].expressions
    let featurePoints = faces[0].featurePoints // each elem contains x and y locations 

    // Draw eye location on canvas

    // Among the feature points
    // Right eye
    // 16 - outer right eye
    // 17 - inner right eye
    // 30 - upper corner right eye
    // 31 - Lower corner right eye

    //Left eye
    // 18 - inner left eye
    // 19 - outer left eye
    // 32 - Upper corner left eye
    // 33 - Lower corner left eye
  }

  socket.emit('next_frame', {'data': faces})
});

/* 
  onImageResults success receives 3 parameters:
  - image: An imageData object containing the pixel values for the processed frame.
  - timestamp: An imageData object contain the pixel values for the processed frame.
  - err_detail: A string contains the encountered exception.
*/
detector.addEventListener("onImageResultsFailure", (image, timestamp, err_detail) => {
  console.log(timestamp, err_detail);
  console.log(image)
});

detector.addEventListener("onResetSuccess", () => {
  console.log("detector was reset successfully");
});

detector.addEventListener("onResetFailure", () => {
  console.log("failed to reset detector");
});

detector.addEventListener("onStopSuccess", () => {
  console.log("Successfully stopped detector")
});

detector.addEventListener("onStopFailure", () => {
  console.log("Failed to stop detector");
});

detector.detectAllExpressions();
detector.detectAllEmotions();
detector.detectAllEmojis();
detector.detectAllAppearance();

// ON PAGE LOAD
$( () => {
  detector.start();
  $(".file-path").val("");

});

