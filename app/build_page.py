"""Fill the Plan Tomorrow page with the latest plan.  Usage: python app/build_page.py"""
from pathlib import Path

plan = Path("reports/plan/plan_tomorrow.json").read_text().replace("</", "<\\/")
page = Path("app/plan_tomorrow_template.html").read_text().replace("__PLAN_JSON__", plan)
Path("app/plan_tomorrow.html").write_text(page)
print("app/plan_tomorrow.html", len(page) // 1024, "KB")
