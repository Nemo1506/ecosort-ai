"""Generates formal, printable ESG Environmental Waste Audit Certificates in HTML.

Self-contained with embedded styling, print media rules, and official seal.
"""
import hashlib
import time


def generate_html_audit_certificate(
    material_label: str,
    display_name: str,
    confidence: float,
    bin_name: str,
    jurisdiction: str,
    steps: list,
    model_name: str,
) -> str:
    """Generate self-contained printable HTML audit certificate."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    raw_hash_seed = f"{material_label}_{confidence}_{timestamp}_{jurisdiction}"
    cert_hash = hashlib.sha256(raw_hash_seed.encode("utf-8")).hexdigest()[:16].upper()
    cert_id = f"ECO-{cert_hash[:4]}-{cert_hash[4:8]}-{cert_hash[8:12]}-{cert_hash[12:]}"

    conf_pct = f"{confidence * 100:.1f}%"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EcoSort AI Formal Waste Audit Certificate - {cert_id}</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    body {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: #09120e;
        color: #f1f5f9;
        margin: 0;
        padding: 40px;
        display: flex;
        justify-content: center;
    }}
    .cert-container {{
        width: 100%;
        max-width: 850px;
        background: linear-gradient(145deg, #0d1e16 0%, #08140f 100%);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 48px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.8);
        position: relative;
    }}
    .cert-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 2px solid rgba(16, 185, 129, 0.3);
        padding-bottom: 24px;
        margin-bottom: 30px;
    }}
    .brand-title {{
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #10b981;
    }}
    .brand-subtitle {{
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .cert-badge {{
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 8px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
    }}
    .meta-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 30px;
    }}
    .meta-item {{
        display: flex;
        flex-direction: column;
    }}
    .meta-label {{
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }}
    .meta-value {{
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
    }}
    .meta-value.accent {{
        color: #34d399;
    }}
    .protocol-box {{
        background: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 30px;
    }}
    .protocol-title {{
        font-size: 14px;
        font-weight: 700;
        color: #34d399;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
    }}
    .protocol-list {{
        margin: 0;
        padding-left: 20px;
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.6;
    }}
    .cert-footer {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid rgba(255,255,255,0.1);
        padding-top: 20px;
        font-size: 12px;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }}
    .seal-box {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .seal-circle {{
        width: 48px;
        height: 48px;
        border: 2px dashed #10b981;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #10b981;
        font-weight: 800;
        font-size: 10px;
        text-align: center;
    }}
    @media print {{
        body {{
            background: #ffffff !important;
            color: #000000 !important;
            padding: 0;
        }}
        .cert-container {{
            background: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #000000 !important;
            box-shadow: none !important;
            max-width: 100% !important;
        }}
        .meta-value, .brand-title {{
            color: #000000 !important;
        }}
        .meta-value.accent {{
            color: #047857 !important;
        }}
        .meta-grid, .protocol-box {{
            background: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
        }}
        .protocol-list {{
            color: #334155 !important;
        }}
    }}
</style>
</head>
<body>
<div class="cert-container">
    <div class="cert-header">
        <div>
            <div class="brand-title">ECOSORT AI™</div>
            <div class="brand-subtitle">Autonomous Material Verification & Circular ESG Audit</div>
        </div>
        <div class="cert-badge">VERIFIED CERTIFICATE</div>
    </div>

    <div class="meta-grid">
        <div class="meta-item">
            <span class="meta-label">Audit Certificate ID</span>
            <span class="meta-value accent">{cert_id}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Verification Timestamp</span>
            <span class="meta-value">{timestamp}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Classified Material Stream</span>
            <span class="meta-value">{display_name} ({material_label})</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Neural Certainty Rating</span>
            <span class="meta-value accent">{conf_pct} (MobileNetV2)</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Designated Municipal Container</span>
            <span class="meta-value">{bin_name}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Regulatory Jurisdiction</span>
            <span class="meta-value">{jurisdiction}</span>
        </div>
    </div>

    <div class="protocol-box">
        <div class="protocol-title">Prescribed Municipal Handling Protocol</div>
        <ul class="protocol-list">
            {"".join(f"<li>{s}</li>" for s in steps)}
        </ul>
    </div>

    <div class="cert-footer">
        <div class="seal-box">
            <div class="seal-circle">ESG<br>PASS</div>
            <div>
                <div><b>EcoSort Intelligence Engine v2.4</b></div>
                <div>Runtime: {model_name} &bull; ImageNet Weights</div>
            </div>
        </div>
        <div>
            <div>Cryptographic Verification Hash:</div>
            <div style="color:#10b981;">{cert_hash}</div>
        </div>
    </div>
</div>
</body>
</html>"""
    return html
