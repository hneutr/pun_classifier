var express = require('express');
var app = express();
var path = require('path');

var port = process.env.PORT || 8080;

app.use('/skeleton', express.static(path.join(__dirname, 'node_modules/skeleton-css/css')));
app.use('/public', express.static(path.join(__dirname, 'public')));

// viewed at http://localhost:8080
app.get('/', function(req, res) {
    res.sendFile(path.join(__dirname + '/index.html'));
});

app.listen(port);
