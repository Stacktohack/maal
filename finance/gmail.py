import smtplib

def send_mail(content):

    mail = smtplib.SMTP('smtp.gmail.com', 587)

    mail.ehlo()

    mail.starttls()

    mail.login('prashantsalerts@gmail.com', 'Alerting')

    print "Content - ", content

    if content is not None:
        print "****sending mail*****"
        mail.sendmail('prashantsalerts@gmail.com', 'prashantrdsoo@gmail.com', content)
        print "sent"
    else:
        print "No mail sent"

    mail.close()