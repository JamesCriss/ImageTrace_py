document.addEventListener('DOMContentLoaded', function() {
    // Get references to the buttons
    const button1 = document.getElementById('button1');
    const button2 = document.getElementById('button2');
    const button3 = document.getElementById('button3');

    // Add event listeners to the buttons
    button1.addEventListener('click', function() {
        console.log('Button 1 clicked');
        window.pywebview.api.button1_clicked();
        // Add your button 1 logic here
    });

    button2.addEventListener('click', function() {
        console.log('Button 2 clicked');
        // Add your button 2 logic here
    });

    button3.addEventListener('click', function() {
        console.log('Button 3 clicked');
        // Add your button 3 logic here
    });
});