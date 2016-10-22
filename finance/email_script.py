# Import smtplib for the actual sending function
import smtplib
import email
# Import the email modules we'll need
from email_script.mime.text import MIMEText

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.

textfile = "email_file.txt"
fp = open(textfile, 'rb')
# Create a text/plain message
msg = MIMEText(fp.read())
fp.close()

me = "prashantrdsoo@gmail.com"
you = "prashantrdsoo@gmail.com"
msg['Subject'] = 'The contents of %s' % textfile
msg['From'] = me
msg['To'] = you

# Send the message via our own SMTP server, but don't include the
# envelope header.
s = smtplib.SMTP('localhost')
s.sendmail(me, [you], msg.as_string())
s.quit()