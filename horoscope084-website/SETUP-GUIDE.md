# Horoscope084.in — One-Page Website Setup Guide (WordPress)

Everything (menu, Services, Courses, About, Contact, Terms, Privacy) lives on **one landing page**.
The menu links jump to each section (`#services`, `#courses`, ...).

## 1. Install plugins
WordPress Dashboard → **Plugins → Add New**, install and activate:
- **Fluent Forms** (service booking + contact forms)
- **Tutor LMS** (courses)
- Optional: a lightweight theme like **Astra**, **Kadence** or **Hello Elementor**

## 2. Create the two forms (Fluent Forms → New Form)
**Form 1 – Service Booking:** Full Name, Email, Phone/WhatsApp, Service (dropdown: Kundli Reading,
Marriage Matching, Career & Finance, Health & Remedies, Numerology, Vastu), Date of Birth,
Time of Birth, Place of Birth, Your Question (textarea). Optional: add a Razorpay/PayPal payment field.

**Form 2 – Contact:** Name, Email, Subject, Message.

Note each form's **ID** (shown in the shortcode `[fluentform id="X"]`).

## 3. Create courses (Tutor LMS → Courses → Add New)
Add your courses (lessons, quizzes, price). Tutor LMS creates a `/courses/` page automatically.

## 4. Create the landing page
1. **Pages → Add New**, title it `Home`.
2. In page settings choose a **Full Width / No Sidebar / Blank** template and disable the page title if your theme allows.
3. Add a **Custom HTML** block and paste the entire contents of `landing-page.html`.
4. Replace everything marked ✏️:
   - `[fluentform id="1"]` → your booking form ID
   - `[fluentform id="2"]` → your contact form ID
   - email, WhatsApp number, city, stats numbers
5. Publish.

> If a shortcode shows as plain text instead of a form, split it out: delete that line from the
> HTML and insert a **Shortcode** block at that spot with the same shortcode.

## 5. Make it the homepage
**Settings → Reading → "A static page" → Homepage: Home** → Save.

## 6. Navigation menu
The page already has its own sticky menu (mobile hamburger included). To avoid two menus:
- Either hide the theme header on this page (most themes have a "Disable Header" page option), **or**
- Keep the theme header and remove the `<nav>…</nav>` block from the HTML, then go to
  **Appearance → Menus** and add **Custom Links**:

| Label    | URL          |
|----------|--------------|
| Home     | `/#home`     |
| Services | `/#services` |
| Courses  | `/#courses`  |
| About    | `/#about`    |
| Contact  | `/#contact`  |
| Terms    | `/#terms`    |
| Privacy  | `/#privacy`  |

## 7. Recommended extras
- Install an SSL certificate (https) — required before taking payments.
- Add Google Fonts *Cinzel* + *Poppins* (theme Customizer → Typography) for the intended look.
- Have the Terms & Privacy text reviewed for your exact business (refund rules, payment gateway, city).
