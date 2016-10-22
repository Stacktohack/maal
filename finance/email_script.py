
import smtplib

def mail_content(content):
    sender = 'from@fromdomain.com'
    receivers = ['prashantrdsoo@gmail.com.com']

    message = content

    try:
       smtpObj = smtplib.SMTP('localhost')
       smtpObj.sendmail(sender, receivers, message)
       print "Successfully sent email"
    except smtplib.SMTPException:
       print "Error: unable to send email"