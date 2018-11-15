// import axios from 'axios'
//
// function get_image(signal) {
//     axios.get('127.0.0.1:8000/img', {
//         signal: signal
//     })
//     .then(response => {
//         var reader = new FileReader();
//         var imageElement = document.getElementById('img');
//         reader.readAsDataURL(response.data);
//         reader.onload = function() {
//
//             var imageDataUrl = reader.result;
//             imageElement.setAttribute("src", imageDataUrl);
//         }
//     })
//     .catch(error => {
//
//     })
// }