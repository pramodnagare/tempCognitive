//Getting AWS SDK object.
var aws = require('aws-sdk');

// Getting aws SES service object.
var ses = new aws.SES({
   region: 'us-east-1'
});

exports.handler = function(event, context, callback) {
    console.log("Lambda Triggered by event: ", event);
    var sns_msg = event.Records[0].Sns.Message;
    console.log(sns_msg);
    var msg_body=sns_msg.split("|");
    var email_to=msg_body[0];
    var email_from=msg_body[0];
    console.log(email_to);
    
	var eParams = {
	  Destination: {
		  ToAddresses: [email_to]
	  },
	  Message: {
		  Body: {
			  Html: {
				  Charset: 'UTF-8',
				  Data: '<html><body><b>Hi There, Knife or Gun object detected. Please take an action immediately</b></body></html>'
			  }
		  },
		  Subject: {
			  Data: "Immediate Action required!"
		  }
	  },
	  Source: email_from
	};

	var email = ses.sendEmail(eParams, function(err, data){
		if(err) console.log(err);
		else {
			console.log("Sending Email to user: ");
			console.log("Email Data: " , email);
			console.log("Email sent successfully!")
			context.succeed(event);
		}
	});
}