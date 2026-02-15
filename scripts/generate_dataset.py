"""
Generiše sintetički dataset za phishing detekciju u istom formatu kao raw.csv.
Kolone: ender, receiver, date, subject, body, label, urls
- label 0 = legitimni mail, 1 = phishing
- urls = broj URL-ova u mailu (ceo broj)

Dataset je dizajniran da:
- Legitimni mailovi budu jasno "normalni" (posao, newsletter, lični) bez lažnih
  phishing signala, kako model ne bi klasifikovao obične mailove kao phishing.
- Phishing mailovi sadrže tipične signale (hitnost, lažni bankarski, verify/login, itd.).
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Broj redova (profesor predlaže 20-30k; možeš promijeniti)
NUM_LEGIT = 14000
NUM_PHISHING = 12000
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_dataset.csv"
# Labelna šuma: dio primjera nasumično pogrešan label → model ne može 100%
LABEL_NOISE_RATE = 0.008

# Različiti legitimni domeni (izgledaju normalno)
LEGIT_DOMAINS = [
    "company.com", "corp.org", "university.edu", "gov.rs", "gmail.com", "outlook.com",
    "yahoo.com", "hotmail.com", "office365.com", "business.com", "team.io", "hr.de",
    "newsletter.io", "notifications.amazon.com", "no-reply.github.com", "shop.com"
]

# Tipični phishing/sumnjivi domeni
PHISHING_DOMAINS = [
    "secure-login.tk", "verify-account.xyz", "bank-update.pw", "paypal-verify.cc",
    "apple-id.ga", "microsoft-support.ml", "amazon-alert.tk", "netflix-renew.xyz"
]

# Legitimni predlošci (subject, body stub) – bez hitnosti, bez "verify your account"
LEGIT_TEMPLATES = [
    ("Meeting reminder", "Hi,\n\nThis is a reminder for our meeting tomorrow at 10:00.\n\nBest regards"),
    ("Your weekly digest", "Here is your weekly summary. You have 3 new updates.\n\nView them in the app."),
    ("Project update", "The project is on track. Please find the latest report attached.\n\nRegards, Team"),
    ("Invoice received", "We have received your payment. Thank you for your business.\n\nAccounting Dept"),
    ("Password changed successfully", "Your password was changed. If this was not you, contact support at support@company.com."),
    ("Newsletter: March edition", "Welcome to our March newsletter. This month we cover new features and tips.\n\nUnsubscribe link at the bottom."),
    ("Calendar invite", "You have been invited to: Budget review. Date: next Tuesday. Location: Conference room A."),
    ("Document shared with you", "Someone shared a document with you. Sign in to your account to view it at https://drive.company.com."),
    ("Subscription confirmation", "You are now subscribed to our service. We will send updates to this address."),
    ("Feedback request", "We would love to hear your feedback. Reply to this email or use our feedback form on the website."),
    ("Delivery notification", "Your order has shipped. Track it at https://shipping.company.com/track/12345."),
    ("Team announcement", "Hello team,\n\nWe have a new policy regarding remote work. Please read the attached document.\n\nHR"),
    ("Your report is ready", "The report you requested is ready. Download it from the link in your dashboard."),
    ("Reminder: timesheet due", "Please submit your timesheet by Friday. Log in at https://hr.company.com."),
    ("Welcome to the platform", "Thanks for signing up. Get started by completing your profile. Visit https://app.company.com."),
]

# Granični legitimni – sadrže riječi tipa urgent/verify/account/click u normalnom kontekstu
LEGIT_BORDERLINE_TEMPLATES = [
    ("Urgent: Team meeting tomorrow", "Hi,\n\nUrgent: we have rescheduled the team meeting to tomorrow 9:00. Please confirm your attendance by replying to this email.\n\nHR Team"),
    ("Verify your email address", "Thanks for signing up. Please verify your email address by clicking the link below. This is a one-time step.\n\nhttps://app.company.com/verify-email"),
    ("Your account statement is ready", "Your monthly account statement is ready. Log in to the portal to view it: https://portal.company.com/statements.\n\nFinance"),
    ("Action required: confirm your attendance", "Please confirm your attendance for the workshop by end of day. Click here to respond: https://hr.company.com/rsvp.\n\nBest regards"),
    ("Security update: password policy", "Our company password policy has been updated. You may need to update your password at next login. Use https://login.company.com.\n\nIT Security"),
    ("Reminder: update your profile", "Please update your profile in the system. Click here to access: https://hr.company.com/profile. Required by end of month."),
    ("Your order has been received", "We have received your order. You can track your delivery and update your address at https://shop.company.com/orders.\n\nCustomer Service"),
    ("Please confirm your details", "We need to confirm your contact details for the directory. Reply to this email or update at https://intranet.company.com.\n\nAdmin"),
]
# "Teški" legitimni – puno riječi kao phishing, bez tipičnih legit fraza (reply/HR/IT); bez company.com linka
LEGIT_HARD_TEMPLATES = [
    ("Urgent: verify your identity", "Dear colleague,\n\nUrgent: we need to verify your identity for the new system. Please confirm your account details. Do not share your password with anyone."),
    ("Action required: update your account", "Your account will be locked if you do not update your security settings. Confirm your details. This is a one-time security check."),
    ("Verify your login credentials", "We noticed a login from a new device. Please verify it was you. Enter your password in the portal when prompted."),
    ("Warning: confirm your payment method", "Your payment method could not be verified. Update your account immediately to avoid suspension."),
    ("Account suspended – action required", "Your account has been temporarily suspended. To restore access, confirm your identity. We will send you a secure link."),
    ("Security alert: verify your account", "We detected unusual activity on your account. Verify your account now. Do not click links in other emails."),
]
# Udio legitimnih: obični / granični / teški (teški = dvosmisleni, izgledaju kao phishing)
LEGIT_BORDERLINE_RATIO = 0.20
LEGIT_HARD_RATIO = 0.10

# Phishing predlošci – sa tipičnim frazama (urgent, verify, account, login, click here, itd.)
PHISHING_TEMPLATES = [
    ("URGENT: Verify your account now", "Your account has been suspended. Verify your identity immediately by clicking here: http://secure-login.tk/verify\n\nDo not ignore this message. Your account will be permanently closed in 24 hours."),
    ("Action required: confirm your PayPal", "We noticed unusual activity. Confirm your PayPal account now: https://paypal-verify.cc/confirm\n\nEnter your password and security details to restore access."),
    ("Your bank account has been locked", "Dear customer,\n\nYour bank account is temporarily locked. Click here to unlock: http://bank-update.pw/unlock\n\nUrgent: complete within 12 hours."),
    ("Apple ID verification required", "Your Apple ID has been used from a new device. Verify now: http://apple-id.ga/verify\n\nIf you did not do this, secure your account immediately."),
    ("Microsoft: unusual sign-in activity", "We detected a sign-in from an unknown device. Confirm it was you: https://microsoft-support.ml/confirm\n\nOtherwise your account may be compromised."),
    ("Netflix: update your payment method", "Your payment failed. Update your payment details now to avoid service interruption: http://netflix-renew.xyz/update"),
    ("Amazon: verify your order", "We could not verify your recent order. Click here to confirm your shipping address and payment: http://amazon-alert.tk/order"),
    ("Your password will expire soon", "Your password expires in 24 hours. Click here to renew: http://secure-login.tk/renew\n\nDo not share this link with anyone."),
    ("WARNING: unauthorized access", "We blocked a sign-in attempt. If this was you, verify your identity: https://verify-account.xyz/restore\n\nAct now to secure your account."),
    ("IRS refund: confirm your details", "You are eligible for a tax refund. Confirm your bank details here: http://bank-update.pw/refund\n\nRequired within 48 hours."),
    ("Your package could not be delivered", "Delivery failed. Confirm your address and pay the redelivery fee: http://amazon-alert.tk/delivery\n\nClick immediately to avoid return."),
    ("Account security alert", "Someone tried to access your account. Verify it was you: https://secure-login.tk/alert\n\nIgnore this and your account will be locked."),
    ("Winning notification", "Congratulations! You have won a prize. Claim it now by clicking here and entering your bank details: http://verify-account.xyz/claim"),
    ("Suspended: policy violation", "Your account was suspended due to policy violation. Appeal now: http://secure-login.tk/appeal\n\nYou have 24 hours to respond."),
    ("Confirm your identity", "We need to confirm your identity. Log in with your password here: https://paypal-verify.cc/login\n\nThis is required to prevent fraud."),
]
# Phishing koji izgleda "profesionalno" (manje CAPS, umjereniji ton) → model može pogriješiti
PHISHING_SOFT_TEMPLATES = [
    ("Your account verification is pending", "We need to verify your account. Please complete the steps at the link below. This helps us keep your account secure.\n\nhttps://secure-login.tk/verify"),
    ("Update your payment information", "Your payment method on file could not be processed. Please update your details at your earliest convenience.\n\nhttp://bank-update.pw/update"),
    ("Confirm your subscription", "You have a pending subscription. Confirm your details to activate your account.\n\nhttps://verify-account.xyz/confirm"),
]
PHISHING_SOFT_RATIO = 0.15  # 15% phishing mailova su "mekani" / dvosmisleni

# Dopune za body (da varira tekst)
LEGIT_BODY_SUFFIXES = [
    "", "\n\nBest regards.", "\n\nThank you.", "\n\nKind regards, Support", "\n\n— The Team"
]
PHISHING_BODY_SUFFIXES = [
    "", "\n\nDo not reply to this email.", "\n\nThis is an automated message.", "\n\nAct now!"
]


def random_date(start_year=2020, end_year=2025):
    d = datetime(start_year, 1, 1) + timedelta(
        seconds=random.randint(0, (end_year - start_year) * 365 * 24 * 3600)
    )
    return d.strftime("%a, %d %b %Y %H:%M:%S %z")


def random_legit_sender():
    name = random.choice(["Ana", "Marko", "Jelena", "Support", "HR", "Info", "No-Reply", "Notifications", "Team"])
    domain = random.choice(LEGIT_DOMAINS)
    local = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(5, 12)))
    return f"{name} <{local}@{domain}>"


def random_legit_receiver():
    local = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(4, 10)))
    domain = random.choice(LEGIT_DOMAINS)
    return f"{local}@{domain}"


def random_phishing_sender():
    name = random.choice(["Security", "Support", "Account", "Billing", "Verify", "Service"])
    domain = random.choice(PHISHING_DOMAINS)
    local = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=random.randint(6, 14)))
    return f"{name} <{local}@{domain}>"


def random_phishing_receiver():
    return random_legit_receiver()


def count_urls_in_text(text: str) -> int:
    import re
    return len(re.findall(r"https?://[^\s]+|www\.[^\s]+", text))


def write_row(writer, ender, receiver, date, subject, body, label, urls_count):
    # csv.writer automatski citira polja sa newline/zarezom
    writer.writerow([
        ender or "",
        receiver or "",
        date or "",
        subject or "",
        body or "",
        label,
        urls_count,
    ])


def main():
    random.seed(42)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # Legitimni: obični / granični / teški (teški bez company.com → model griješi → FPR ~1–2%)
    n_hard = int(NUM_LEGIT * LEGIT_HARD_RATIO)
    n_borderline = int(NUM_LEGIT * LEGIT_BORDERLINE_RATIO)
    n_normal = NUM_LEGIT - n_borderline - n_hard
    for i in range(NUM_LEGIT):
        if i < n_hard:
            subj, body = random.choice(LEGIT_HARD_TEMPLATES)
        elif i < n_hard + n_borderline:
            subj, body = random.choice(LEGIT_BORDERLINE_TEMPLATES)
        else:
            subj, body = random.choice(LEGIT_TEMPLATES)
        body = body + random.choice(LEGIT_BODY_SUFFIXES)
        # Ponekad dodaj legitimni link (ne za "teške" legite – oni ostaju bez jakog signala)
        if i >= n_hard and random.random() < 0.3:
            body += "\nhttps://www.company.com/help"
        urls_count = count_urls_in_text(body)
        rows.append({
            "ender": random_legit_sender(),
            "receiver": random_legit_receiver(),
            "date": random_date(),
            "subject": subj,
            "body": body,
            "label": 0,
            "urls": urls_count,
        })
    # Phishing (dio obični, dio "meki" / dvosmisleni)
    n_soft_phishing = int(NUM_PHISHING * PHISHING_SOFT_RATIO)
    for i in range(NUM_PHISHING):
        if i < n_soft_phishing:
            subj, body = random.choice(PHISHING_SOFT_TEMPLATES)
        else:
            subj, body = random.choice(PHISHING_TEMPLATES)
        body = body + random.choice(PHISHING_BODY_SUFFIXES)
        urls_count = count_urls_in_text(body)
        rows.append({
            "ender": random_phishing_sender(),
            "receiver": random_phishing_receiver(),
            "date": random_date(),
            "subject": subj,
            "body": body,
            "label": 1,
            "urls": urls_count,
        })

    random.shuffle(rows)

    # Labelna šuma: dio primjera dobije pogrešan label (dataset nije savršen)
    if LABEL_NOISE_RATE > 0:
        n_noise = max(1, int(len(rows) * LABEL_NOISE_RATE))
        noise_idx = random.sample(range(len(rows)), n_noise)
        for idx in noise_idx:
            rows[idx]["label"] = 1 - rows[idx]["label"]
        print(f"Labelna šuma: {n_noise} primjera s promijenjenim labelom")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ender", "receiver", "date", "subject", "body", "label", "urls"])
        for r in rows:
            write_row(
                writer,
                r["ender"],
                r["receiver"],
                r["date"],
                r["subject"],
                r["body"],
                r["label"],
                r["urls"],
            )

    print(f"Dataset zapisan: {OUTPUT_PATH}")
    print(f"Ukupno redova: {len(rows)}")
    print(f"Legitimni (0): {NUM_LEGIT}, Phishing (1): {NUM_PHISHING}")


if __name__ == "__main__":
    main()
