function AudioController() {

  this.send = function(file) {
  }

  this.createWebSocket = function() {
    var uri = 'ws://localhost:6600';
    var ws = new WebSocket(uri);
