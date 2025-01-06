from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import logging
from typing import Union


def send_email(subject: str, receivers: Union[list[str], str], content: str) -> str:
    """给指定的邮箱发送邮件"""
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["To"] = "; ".join(receivers) if isinstance(
        receivers, list) else receivers
    message["From"] = 'LLM'
    message.attach(MIMEText(content, "html", _charset="utf-8"))

    try:
        with smtplib.SMTP(host="relay.homecredit.cn", port=25) as server:
            server.send_message(message)
        logging.info("邮件发送成功.")
        return '邮件发送成功'
    except Exception as e:
        logging.error(f"邮件发送失败: {e}")
        return '邮件发送失败'


# def send_email(subject: str, receivers: Union[list[str], str], content: str):
#     """给指定的邮箱发送邮件"""
#     message = MIMEMultipart("alternative")
#     message["Subject"] = subject
#     message["To"] = "; ".join(receivers) if isinstance(
#         receivers, list) else receivers
#     message["From"] = "LLM"
#     message.attach(MIMEText(content, "html", _charset="utf-8"))
#     try:
#         with smtplib.SMTP(host="relay.homecredit.cn", port=25) as server:
#             server.sendmail("LLM", receivers, message.as_string())
#         logging.info("邮件发送成功.")
#         return '邮件发送成功'
#     except Exception as e:
#         print(f"邮件发送失败: {e}")
#     else:
#         print("邮件发送成功.")
#     finally:
#         pass   

    

if __name__ == "__main__":
    send_email("", "", "")