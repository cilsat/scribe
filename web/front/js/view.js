var supportedMIMETypes = [
  'audio/x-raw',
  'audio/wav',
  'audio/ogg',
  'audio/webm'
]

var audio = new AudioController();

function toggleMic() {
  if (!$('#buttonMic').hasClass('disabled')) {
    $('#buttonMic').toggleClass('active');

    if ($('#buttonMic').hasClass('active'))) {
      audio.record();
    } else {
      audio.stop();
    }
  }
}

function handleFile() {
  var file = document.getElementById('audioFile').files[0];
  if (file) {
    var info;
    if (supportedMIMETypes.indexOf(file.type) != -1) {
      info = file.name;
    } else {
      info = 'Unsupported format ' + file.type;
    }
    $('#infoFile').html(info);
  } else {
    console.log($('#audioFile').val());
  }
}

function sendFile() {
  audio.send($('#audioFile').get(0).files[0]);
}

