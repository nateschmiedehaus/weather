<?php
	// Get the form data
	$name = $_POST['name'];
	$email = $_POST['email'];
	$message = $_POST['message'];
	
	// Validate the form data
	if (empty($name) || empty($email) || empty($message)) {
		echo "All fields are required!";
		exit;
	}
	
	// Send the email
	$to = "your-email@example.com"; // Change this to your own email address
	$subject = "New contact form submission";
	$body = "Name: $name\nEmail: $email\nMessage: $message";
	$headers = "From: $email";
	
	if (mail($to, $subject, $body, $headers)) {
		echo "Thank you for your message!";
	} else {
		echo "Oops! Something went wrong.";
	}
?>
