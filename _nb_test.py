# ─── Cell 1: Imports & Config ────────────────────────────────────────────────
import pandas as pd
import numpy as np
import random
import os
import pathlib
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# Resolve data dir: works whether notebook is opened from project root or notebooks/
_cwd = pathlib.Path().resolve()
OUTPUT_DIR = str(_cwd / 'data') if (_cwd / 'data').exists() else str(_cwd.parent / 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 18 months back from today
TODAY      = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = TODAY - timedelta(weeks=78)   # ~18 months ago
WEEKS      = 78

week_starts = [START_DATE + timedelta(weeks=i) for i in range(WEEKS)]
week_labels = [d.strftime('%Y-W%W') for d in week_starts]

MARKETS = ['India', 'US']
BRANDS  = ["Lay's", 'Pringles', 'Doritos']

# ── Flavors for ALL brands (used across ALL datasets)
FLAVORS = {
    "Lay's": [
        'Classic Salted',
        'Magic Masala',
        'Spanish Tomato Tango',
        'American Style Cream & Onion',
        'Chile Limon',
        'Wafer Style',
        'Baked',
        'Max Peri Peri',
        'Herb & Onion',
        'World Cup Special Edition',   # appears from week ~55 onwards
    ],
    'Pringles': [
        'Original',
        'Sour Cream & Onion',
        'BBQ',
        'Pizza',
        'Wavy Classic',
        'Wavy Ranch',
        'Cheddar Cheese',
    ],
    'Doritos': [
        'Nacho Cheese',
        'Cool Ranch',
        'Spicy Nacho',
        'Flamin Hot',
        'Sweet Chili',
        'Flamin Hot Lime',             # new launch around week ~50
    ]
}

# ── 6 Anomaly events spread across 78 weeks
ANOMALIES = {
    8:  {'type': 'external',      'desc': 'Health report links ultra-processed snacks to obesity — category-wide hit'},
    18: {'type': 'lays_crisis',   'desc': "Lay's India packaging controversy goes viral — foreign object found"},
    30: {'type': 'competitor',    'desc': 'Pringles launches Wavy line + aggressive 20% price cut India & US'},
    42: {'type': 'competitor',    'desc': 'Doritos influencer scandal — collateral trust impact on snack category'},
    55: {'type': 'lays_positive', 'desc': "Lay's World Cup Special Edition launch — massive buzz"},
    68: {'type': 'lays_crisis',   'desc': "Lay's US supply chain disruption — out of stock in 3 major retailers"},
}

def get_flavor(brand, week_idx):
    """Return a flavor for a brand, respecting launch timelines."""
    pool = FLAVORS[brand].copy()
    # World Cup Edition only available from week 55
    if brand == "Lay's" and week_idx < 55:
        pool = [f for f in pool if f != 'World Cup Special Edition']
    # Wavy range only from week 28 (Pringles launch)
    if brand == 'Pringles' and week_idx < 28:
        pool = [f for f in pool if 'Wavy' not in f]
    # Flamin Hot Lime only from week 48
    if brand == 'Doritos' and week_idx < 48:
        pool = [f for f in pool if f != 'Flamin Hot Lime']
    return random.choice(pool) if pool else random.choice(FLAVORS[brand])

_lays = "Lay's"
print('✅ Config ready.')
print(f'   Period : {START_DATE.strftime("%Y-%m-%d")} → {TODAY.strftime("%Y-%m-%d")} ({WEEKS} weeks)')
print(f'   Output : {OUTPUT_DIR}')
print(f'   Brands : {BRANDS}')
print(f"   Flavors: Lay's={len(FLAVORS[_lays])}, Pringles={len(FLAVORS['Pringles'])}, Doritos={len(FLAVORS['Doritos'])}")
print(f'   Anomaly weeks: {list(ANOMALIES.keys())}')
# ─── Cell 2: Helper Utilities ────────────────────────────────────────────────

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def seasonal_multiplier(week_idx, market):
    """78-week seasonal curve — repeats annual events across 18 months."""
    # Normalize to position within year (0-51)
    week_in_year = week_idx % 52
    if market == 'India':
        republic = np.exp(-0.5 * ((week_in_year - 2)  / 1.5) ** 2) * 0.15
        holi     = np.exp(-0.5 * ((week_in_year - 9)  / 1.5) ** 2) * 0.20
        ipl      = np.exp(-0.5 * ((week_in_year - 14) / 4)   ** 2) * 0.35
        diwali   = np.exp(-0.5 * ((week_in_year - 42) / 2)   ** 2) * 0.45
        christmas= np.exp(-0.5 * ((week_in_year - 50) / 2)   ** 2) * 0.20
        return 1.0 + republic + holi + ipl + diwali + christmas
    else:  # US
        superbowl = np.exp(-0.5 * ((week_in_year - 5)  / 2)   ** 2) * 0.40
        march_mad = np.exp(-0.5 * ((week_in_year - 11) / 2)   ** 2) * 0.20
        memorial  = np.exp(-0.5 * ((week_in_year - 21) / 1.5) ** 2) * 0.25
        summer    = np.exp(-0.5 * ((week_in_year - 28) / 4)   ** 2) * 0.30
        thanksgiving = np.exp(-0.5*((week_in_year - 47) / 2)  ** 2) * 0.35
        return 1.0 + superbowl + march_mad + memorial + summer + thanksgiving

def anomaly_sentiment_delta(week_idx, market):
    """Returns (sentiment_delta, volume_multiplier) for Lay's."""
    sent  = 0.0
    vol   = 1.0
    # Health report
    if week_idx in [8, 9]:    sent -= 0.08; vol *= 1.30
    # Lays India packaging crisis
    if week_idx == 18 and market == 'India': sent -= 0.32; vol *= 3.20
    if week_idx == 19 and market == 'India': sent -= 0.18; vol *= 2.00
    if week_idx == 20 and market == 'India': sent -= 0.08; vol *= 1.40
    # Pringles price cut
    if week_idx in [30, 31]:  sent -= 0.05; vol *= 1.15
    # Doritos controversy
    if week_idx == 42:        sent -= 0.06; vol *= 1.20
    if week_idx == 43:        sent -= 0.03; vol *= 1.10
    # World Cup launch — positive
    if week_idx in [55, 56]:  sent += 0.18; vol *= 2.50
    if week_idx == 57:        sent += 0.10; vol *= 1.60
    # US supply chain crisis
    if week_idx == 68 and market == 'US': sent -= 0.22; vol *= 2.00
    if week_idx == 69 and market == 'US': sent -= 0.12; vol *= 1.50
    if week_idx == 70 and market == 'US': sent -= 0.05; vol *= 1.20
    return sent, vol

print('✅ Helpers ready.')
# ─── Cell 3: Dataset 1 — Social Posts ────────────────────────────────────────

PLATFORMS = ['Twitter/X', 'Instagram', 'Reddit']
PLATFORM_PROFILES = {
    'Twitter/X': {'likes': (5, 800),   'share_ratio': (0.05, 0.35), 'reply_ratio': (0.02, 0.12)},
    'Instagram': {'likes': (40, 4500), 'share_ratio': (0.01, 0.08), 'reply_ratio': (0.01, 0.06)},
    'Reddit':    {'likes': (1, 2200),  'share_ratio': (0.00, 0.00), 'reply_ratio': (0.10, 0.60)},
}

TEMPLATES = {
    "Lay's": {
        'India': {
            'positive': [
                "Lay's {flavor} ke bina movie dekhna possible hi nahi yaar 🍿😂 #{tag}",
                "Bhai Lay's {flavor} try karo ek baar, life badal jayegi! 🔥 #Lays #{tag}",
                "IPL + Lay's {flavor} = perfect combo 🏏❤️ #{tag}",
                "Lay's {flavor} wala packet khatam hi nahi hota 😭 #snacklover",
                "Just had Lay's {flavor} — still the GOAT of chips 👑 #{tag}",
                "Ghar pe friends aaye the, Lay's {flavor} ne sabko khush kar diya 😄 #{tag}",
                "Lay's {flavor} is my comfort food fr fr 🥺💯 #Lays",
                "Ek packet {flavor} liya tha, ab teen ho gaye 😅 #{tag} #addicted",
            ],
            'negative': [
                "Yaar @Lays ka {flavor} packet 80% hawa hai 😤 #{tag}",
                "Lay's {flavor} ka taste pehle jaisa nahi raha. Quality gir gayi 😒",
                "₹20 ka {flavor} liya, 8 chips mile. @Lays ye kya mazaak hai?? #{tag}",
                "Lay's {flavor} expired nikla. Third time this month. Done 🤬",
                "@PepsiCo @Lays price badhao aur quantity ghataao 😡 #{tag}",
            ],
            'neutral': [
                "Lay's {flavor} ya Kurkure — aaj ka sawaal 🤔 #{tag}",
                "Dekha Lay's ka naya {flavor} ad, interesting hai #{tag}",
                "Lay's {flavor} discount mein hai BigBazaar mein this week 🛒 #{tag}",
                "Koi batao Lay's {flavor} vs {flavor2} mein kaun better hai? #{tag}",
            ],
            'controversy': [
                "YE KYA HAI MERE LAY'S {flavor} PACKET MEIN?? 🤢 @Lays @PepsiCo #boycottLays #{tag}",
                "Thread 🧵: Lay's India {flavor} packaging mein jo mila wo dekhke pet kharab ho gaya",
                "Itne bade brand se ye umeed nahi thi. @Lays {flavor} ka QC kahan gaya?? #laysgate",
                "Dosto PLEASE Lay's {flavor} mat kharido abhi. Serious issue hai 😱 #boycottLays",
                "@Lays {flavor} packet mein jo tha uski photo share kar raha hoon. RT karo 🔁 #{tag}",
            ],
            'worldcup_buzz': [
                "Lay's {flavor} World Cup Edition mil gayi bhai!! Limited hai, jao jao 🏆🔥 #{tag}",
                "Lay's ka World Cup {flavor} is INSANE 🤯 perfect timing @Lays #{tag}",
                "World Cup dekhte waqt Lay's {flavor} — life set hai yaar 🏏🥔 #{tag}",
                "Bhai Lay's {flavor} World Cup packet collect kar raha hoon 😂 #{tag}",
            ],
            'supply_crisis': [
                "Lay's {flavor} kahi nahi mil raha! Kya hua @Lays?? #{tag}",
                "Teen stores gaya, {flavor} sold out everywhere. @Lays stock issue? #{tag}",
            ]
        },
        'US': {
            'positive': [
                "Lay's {flavor} hits different on game day fr 🏈🔥 #{tag}",
                "Can't watch the game without Lay's {flavor} — it's tradition 🏈",
                "Lay's {flavor} with French onion dip is criminally underrated 👇 #{tag}",
                "Grabbed Lay's {flavor} for the road trip — perfect as always ✅ #{tag}",
                "My kids won't eat any other chips. Lay's {flavor} for life 😄 #{tag}",
                "Lay's {flavor} is consistently the best chip, no debate #{tag}",
                "Just discovered Lay's {flavor} and where has this been all my life 😭",
            ],
            'negative': [
                "Why does Lay's {flavor} taste like cardboard now?? Changed the recipe 😤",
                "Lay's bags are literally 60% air. Scam. #{tag} #ripoff",
                "Price went up AGAIN on Lay's {flavor}. Switching to store brand #{tag}",
                "Found Lay's {flavor} stale 2 weeks before expiry. Done. #{tag}",
                "@Lays the {flavor} has gotten SO salty recently. Ruined it #{tag}",
            ],
            'neutral': [
                "Lay's {flavor} vs {flavor2} — which for dipping? 🤔 #{tag}",
                "Saw the new Lay's {flavor} ad. Pretty solid campaign #{tag}",
                "Lay's {flavor} on sale at Walmart this week 🛒 #{tag}",
                "Does anyone actually like Lay's {flavor}? Asking for a friend #{tag}",
            ],
            'worldcup_buzz': [
                "Lay's {flavor} World Cup Edition is actually so good 🏆 #{tag}",
                "Got the limited Lay's {flavor} World Cup pack!! 🔥 #{tag}",
                "Lay's x World Cup collab is lowkey amazing #{tag}",
            ],
            'supply_crisis': [
                "Can't find Lay's {flavor} ANYWHERE — 4 stores checked. @Lays what's going on?? #{tag}",
                "Lay's {flavor} out of stock at Walmart, Target AND Kroger. Supply issue?? #{tag}",
                "Been 2 weeks without Lay's {flavor} on shelves near me. Really @Lays?? #{tag}",
            ]
        }
    },
    'Pringles': {
        'India': {
            'positive': [
                "Pringles {flavor} is so much better value now after price cut 👌 #{tag}",
                "Pringles {flavor} ka canister perfect hai road trips ke liye 🚗 #{tag}",
                "Pringles {flavor} ne Lay's ki jagah le li mere liye. Quality consistent hai #{tag}",
            ],
            'negative': [
                "Pringles {flavor} bahut artificial lagta hai. Natural taste nahi #{tag}",
                "Pringles {flavor} India mein nahi mil raha #{tag} #stockissue",
            ],
            'neutral': [
                "Pringles {flavor} vs Lay's {flavor2} — aaj ka debate 🤔 #{tag}",
                "Pringles ne naya {flavor} launch kiya. Try karna padega #{tag}",
            ]
        },
        'US': {
            'positive': [
                "Once you pop you cannot stop — Pringles {flavor} is too good 😭 #{tag}",
                "Pringles {flavor} is the move for parties. No mess 🙌 #{tag}",
                "Pringles dropped prices AND launched {flavor}?? Best week ever #{tag}",
            ],
            'negative': [
                "Pringles {flavor} tastes more artificial than Lay's. Hard pass #{tag}",
                "Can't find Pringles {flavor} near me. Supply issue? #{tag}",
            ],
            'neutral': [
                "Pringles {flavor} vs Lay's {flavor2} — who wins? #{tag}",
                "Saw Pringles' new {flavor} at the store. Might grab it #{tag}",
            ]
        }
    },
    'Doritos': {
        'India': {
            'positive': [
                "Doritos {flavor} with salsa dip is another level yaar 🌮🔥 #{tag}",
                "Gaming session ke liye Doritos {flavor} best hai 🎮 #{tag}",
            ],
            'negative': [
                "Doritos influencer wala controversy dekha? Brand trust khatam 😤 #{tag}",
                "Doritos {flavor} bahut zyada salt hai. Dobara nahi lunga #{tag}",
            ],
            'neutral': [
                "Doritos {flavor} is okay. Lay's still better for me #{tag}",
                "Doritos ka naya {flavor} ad dekha. Campaign theek tha #{tag}",
            ]
        },
        'US': {
            'positive': [
                "Doritos {flavor} is the perfect gaming snack no cap 🎮 #{tag}",
                "Doritos {flavor} with guac is elite. No further questions #{tag}",
            ],
            'negative': [
                "That Doritos influencer situation was so tone deaf 😤 #{tag}",
                "Doritos {flavor} way too salty. Lay's all day #{tag}",
                "Cancelled Doritos after that influencer drama 😤 #{tag}",
            ],
            'neutral': [
                "Doritos {flavor} is decent but Lay's hits different #{tag}",
                "New Doritos {flavor} spotted at Target. Anyone tried it? #{tag}",
            ]
        }
    }
}

MARKET_TAGS = {'India': 'IndiaSnacks', 'US': 'USSnacks'}
BASE_POSTS  = {'India': 90, 'US': 75}

rows = []
post_id = 1

for w_idx, (w_start, w_label) in enumerate(zip(week_starts, week_labels)):
    for market in MARKETS:
        sm = seasonal_multiplier(w_idx, market)
        for brand in BRANDS:
            sent_delta, vol_mult = anomaly_sentiment_delta(w_idx, market) if brand == "Lay's" else (0, 1.0)

            # Competitor spikes
            if brand == 'Pringles' and w_idx in range(28, 35): vol_mult = max(vol_mult, 1.8)
            if brand == 'Doritos'  and w_idx in [42, 43]:      vol_mult = max(vol_mult, 2.2)

            count = clamp(int(BASE_POSTS[market] * sm * vol_mult * np.random.uniform(0.82, 1.18)), 10, 2500)

            # Sentiment weights
            if brand == "Lay's":
                pos_w = clamp(0.56 + sent_delta, 0.08, 0.85)
                neg_w = clamp(0.14 - sent_delta * 0.5, 0.08, 0.75)
            elif brand == 'Pringles':
                pos_w = clamp(0.48 + (0.12 if w_idx in range(28,35) else 0), 0.20, 0.80)
                neg_w = 0.22
            else:
                pos_w = clamp(0.46 - (0.20 if w_idx in [42,43] else 0), 0.15, 0.75)
                neg_w = clamp(0.24 + (0.25 if w_idx in [42,43] else 0), 0.15, 0.65)
            neu_w = clamp(1 - pos_w - neg_w, 0.05, 0.40)

            sentiments = random.choices(['positive','negative','neutral'],
                                        weights=[pos_w, neg_w, neu_w], k=count)

            tmpl = TEMPLATES[brand][market]

            for sent in sentiments:
                platform = random.choices(
                    PLATFORMS,
                    weights=[0.40,0.40,0.20] if market=='India' else [0.35,0.35,0.30]
                )[0]

                flavor  = get_flavor(brand, w_idx)
                flavor2 = get_flavor(brand, w_idx)
                tag     = MARKET_TAGS[market]

                # Pick template pool based on anomaly context
                if brand=="Lay's" and w_idx==18 and market=='India' and sent=='negative':
                    pool = tmpl.get('controversy', tmpl[sent])
                elif brand=="Lay's" and w_idx in [55,56,57] and sent=='positive':
                    pool = tmpl.get('worldcup_buzz', tmpl[sent])
                elif brand=="Lay's" and w_idx in [68,69,70] and market=='US' and sent=='negative':
                    pool = tmpl.get('supply_crisis', tmpl[sent])
                else:
                    pool = tmpl.get(sent, tmpl['neutral'])

                text = random.choice(pool).format(flavor=flavor, flavor2=flavor2, tag=tag)

                # Engagement
                prof  = PLATFORM_PROFILES[platform]
                likes = clamp(
                    int(np.random.lognormal(np.log(np.sqrt(prof['likes'][0]*prof['likes'][1])), 0.9)),
                    prof['likes'][0], prof['likes'][1]*3
                )
                if w_idx==18 and brand=="Lay's" and market=='India': likes=int(likes*np.random.uniform(4,12))
                if w_idx in [55,56] and brand=="Lay's":              likes=int(likes*np.random.uniform(2,5))
                if w_idx in [68,69] and brand=="Lay's" and market=='US': likes=int(likes*np.random.uniform(2,6))

                shares   = int(likes*np.random.uniform(*prof['share_ratio'])) if platform!='Reddit' else 0
                comments = int(likes*np.random.uniform(*prof['reply_ratio']))
                post_date = w_start + timedelta(days=random.randint(0,6), hours=random.randint(0,23))

                rows.append({
                    'post_id':          f'P{post_id:07d}',
                    'week':             w_label,
                    'week_start_date':  w_start.strftime('%Y-%m-%d'),
                    'post_date':        post_date.strftime('%Y-%m-%d'),
                    'post_hour':        post_date.hour,
                    'market':           market,
                    'platform':         platform,
                    'brand_mentioned':  brand,
                    'flavor':           flavor,
                    'sentiment':        sent,
                    'text':             text,
                    'likes':            likes,
                    'shares':           shares,
                    'comments':         comments,
                    'total_engagement': likes+shares+comments,
                    'is_anomaly_week':  w_idx in ANOMALIES,
                    'anomaly_type':     ANOMALIES.get(w_idx,{}).get('type',''),
                })
                post_id += 1

social_df = pd.DataFrame(rows)
social_df.to_csv(f'{OUTPUT_DIR}/social_posts.csv', index=False)

lays_f     = sorted(social_df[social_df['brand_mentioned'] == "Lay's"]['flavor'].unique())
pringles_f = sorted(social_df[social_df['brand_mentioned'] == 'Pringles']['flavor'].unique())
doritos_f  = sorted(social_df[social_df['brand_mentioned'] == 'Doritos']['flavor'].unique())
print(f'✅ social_posts.csv → {len(social_df):,} rows')
print(f"   Lay's flavors in data   : {lays_f}")
print(f'   Pringles flavors in data : {pringles_f}')
print(f'   Doritos flavors in data  : {doritos_f}')
# ─── Cell 4: Dataset 2 — Search Trends ───────────────────────────────────────

KEYWORDS = {
    "Lay's": {
        'India': {
            'lays chips':                        74000,
            'lays magic masala':                 51000,
            'lays classic salted':               38000,
            'lays american cream onion':         29000,
            'lays spanish tomato tango':         22000,
            'lays max peri peri':                18000,
            'lays world cup edition':             5000,  # activates week 55
            'buy lays online':                   31000,
            'lays price':                        28000,
        },
        'US': {
            'lays chips':                        68000,
            'lays classic':                      55000,
            'lays sour cream onion':             29000,
            'lays baked':                        24000,
            'lays chile limon':                  19000,
            'lays world cup edition':             4000,  # activates week 55
            'buy lays bulk':                     18000,
            'lays out of stock':                  3000,  # activates week 68
        }
    },
    'Pringles': {
        'India': {
            'pringles chips':                    38000,
            'pringles original':                 22000,
            'pringles sour cream onion':         18000,
            'pringles wavy':                      9000,  # activates week 28
            'pringles price india':              16000,
            'pringles cheddar':                  11000,
        },
        'US': {
            'pringles chips':                    62000,
            'pringles sour cream':               34000,
            'pringles original':                 28000,
            'pringles bbq':                      21000,
            'pringles wavy':                     14000,  # activates week 28
            'pringles price drop':               11000,
        }
    },
    'Doritos': {
        'India': {
            'doritos nacho cheese':              29000,
            'doritos cool ranch':                17000,
            'doritos flamin hot':                21000,
            'doritos flamin hot lime':            8000,  # activates week 48
            'doritos sweet chili':               12000,
            'doritos controversy':                4000,
        },
        'US': {
            'doritos nacho cheese':              58000,
            'doritos cool ranch':                48000,
            'doritos flamin hot':                42000,
            'doritos flamin hot lime':           15000,  # activates week 48
            'doritos spicy nacho':               31000,
            'doritos controversy':                8000,
        }
    },
    'category': {
        'India': {
            'best chips brand india':            48000,
            'healthy snacks india':              71000,
            'chips on offer today':              39000,
            'snacks for ipl party':              22000,
        },
        'US': {
            'best chips brand':                  52000,
            'healthy chip snacks':               88000,
            'chips on sale near me':             63000,
            'super bowl snack ideas':            44000,
        }
    }
}

rows = []
for w_idx, (w_start, w_label) in enumerate(zip(week_starts, week_labels)):
    for market in MARKETS:
        sm = seasonal_multiplier(w_idx, market)
        for brand_key, market_kws in KEYWORDS.items():
            for kw, base_vol in market_kws.get(market, {}).items():

                # Flavor launch gating
                if 'wavy' in kw          and w_idx < 28: continue
                if 'world cup' in kw     and w_idx < 55: continue
                if 'flamin hot lime' in kw and w_idx < 48: continue
                if 'out of stock' in kw  and w_idx not in range(68,74): continue

                vol = int(base_vol * sm * np.random.uniform(0.85, 1.15))

                # Anomaly boosts
                if 'lays' in kw:
                    if w_idx in [18,19,20] and market=='India': vol=int(vol*np.random.uniform(2.8,4.2))
                    if w_idx in [55,56,57]:                     vol=int(vol*np.random.uniform(2.0,3.0))
                    if w_idx in [68,69,70] and market=='US':    vol=int(vol*np.random.uniform(2.5,4.0))
                if 'pringles' in kw and w_idx in range(28,36):  vol=int(vol*np.random.uniform(1.9,2.6))
                if 'doritos' in kw and w_idx in [42,43]:        vol=int(vol*np.random.uniform(2.2,3.1))
                if 'controversy' in kw:
                    vol = int(vol*(np.random.uniform(8,15) if w_idx in [42,43] else np.random.uniform(0.2,0.5)))
                if 'ipl' in kw and (w_idx%52) in range(12,18): vol=int(vol*np.random.uniform(2.5,3.5))

                # Extract flavor from keyword
                flavor_in_kw = kw.replace(brand_key.lower().replace("lay's",'lays'),'').strip()

                rows.append({
                    'week':            w_label,
                    'week_start_date': w_start.strftime('%Y-%m-%d'),
                    'market':          market,
                    'brand':           brand_key,
                    'keyword':         kw,
                    'flavor':          flavor_in_kw if flavor_in_kw else 'brand_generic',
                    'search_volume':   vol,
                    'relative_index':  round(vol/base_vol,3),
                    'yoy_change_pct':  round(np.random.normal(8,12),1),
                    'is_anomaly_week': w_idx in ANOMALIES,
                    'anomaly_type':    ANOMALIES.get(w_idx,{}).get('type',''),
                })

search_df = pd.DataFrame(rows)
search_df.to_csv(f'{OUTPUT_DIR}/search_trends.csv', index=False)
print(f'✅ search_trends.csv → {len(search_df):,} rows')
for b in ["Lay's", 'Pringles', 'Doritos']:
    flavors = search_df[search_df['brand']==b]['flavor'].unique().tolist()
    print(f'   {b} flavor keywords: {flavors}')
# ─── Cell 5: Dataset 3 — Campaign Metrics ────────────────────────────────────

# (name, market, start_week, duration, type, budget_INR_lakhs, primary_flavor)
CAMPAIGNS = [
    # ── Year 1 (weeks 0-51)
    ('Lay\'s Diwali Masti',            'India',  0,  3, 'TV',         320, 'Magic Masala'),
    ('Lay\'s New Year Digital',        'India',  3,  2, 'Digital',     55, 'Classic Salted'),
    ('Lay\'s Holi Colour Crunch',      'India',  8,  2, 'TV',         140, 'Spanish Tomato Tango'),
    ('Lay\'s IPL Season Anthem',       'India', 14,  5, 'TV',         280, 'Magic Masala'),
    ('Lay\'s IPL Influencer Blitz',    'India', 14,  3, 'Influencer',  65, 'Max Peri Peri'),
    ('Lay\'s Independence Day Push',   'India', 37,  2, 'Digital',     48, 'Classic Salted'),
    ('Lay\'s Diwali 2025 Campaign',    'India', 43,  3, 'TV',         350, 'Magic Masala'),
    ('Lay\'s Diwali Influencer',       'India', 43,  2, 'Influencer',  70, 'American Style Cream & Onion'),
    # ── US Year 1
    ('Lay\'s Super Bowl Party Pack',   'US',     4,  2, 'TV',         420, 'Classic'),
    ('Lay\'s Super Bowl Digital',      'US',     4,  2, 'Digital',    110, 'Sour Cream & Onion'),
    ('Lay\'s March Madness',           'US',    10,  3, 'TV',         230, 'Classic'),
    ('Lay\'s Summer Creator Series',   'US',    22,  3, 'Influencer',  75, 'Chile Limon'),
    ('Lay\'s Labor Day Grilling',      'US',    38,  2, 'TV',         195, 'Classic'),
    ('Lay\'s Thanksgiving Snack Up',   'US',    47,  2, 'TV',         260, 'Classic'),
    # ── World Cup special (weeks 55-60)
    ('Lay\'s World Cup Edition Launch','India', 55,  4, 'TV',         400, 'World Cup Special Edition'),
    ('Lay\'s WC Creator Collab',       'India', 55,  3, 'Influencer',  90, 'World Cup Special Edition'),
    ('Lay\'s WC US Digital',           'US',    55,  3, 'Digital',    120, 'World Cup Special Edition'),
    # ── Year 2 wrap (weeks 52-77)
    ('Lay\'s New Year 2026',           'India', 52,  2, 'Digital',     60, 'Magic Masala'),
    ('Lay\'s IPL 2026 Anthem',         'India', 64,  4, 'TV',         300, 'Magic Masala'),
    ('Lay\'s Super Bowl 2026',         'US',    56,  2, 'TV',         450, 'Classic'),
]

BENCHMARKS = {
    'TV':         {'India':{'cpm':180, 'ctr':0.000,'cvr':0.000},
                   'US':   {'cpm':28,  'ctr':0.000,'cvr':0.000}},
    'Digital':    {'India':{'cpm':95,  'ctr':0.028,'cvr':0.062},
                   'US':   {'cpm':14,  'ctr':0.021,'cvr':0.055}},
    'Influencer': {'India':{'cpm':55,  'ctr':0.048,'cvr':0.088},
                   'US':   {'cpm':22,  'ctr':0.038,'cvr':0.071}},
}

rows = []
for camp in CAMPAIGNS:
    name, market, start_w, dur, ctype, budget_lakh, flavor = camp
    budget_inr = budget_lakh * 100000

    for wi in range(start_w, min(start_w+dur, WEEKS)):
        wic  = wi - start_w
        sm   = seasonal_multiplier(wi, market)
        decay= max(0.65, 1.0 - 0.12*wic)
        bm   = BENCHMARKS[ctype][market]

        spend_inr   = int(budget_inr/dur*decay*np.random.uniform(0.92,1.08))
        impressions = int((spend_inr/bm['cpm'])*1000*sm*np.random.uniform(0.88,1.12))
        reach       = int(impressions*np.random.uniform(0.42,0.68))

        if ctype=='TV':
            clicks=0; ctr=0.0; conversions=0
            brand_lift=round(np.random.uniform(0.025,0.075)*sm*decay,4)
            view_rate=round(np.random.uniform(0.55,0.82),3)
        else:
            ctr=round(bm['ctr']*np.random.uniform(0.75,1.35),4)
            clicks=int(impressions*ctr)
            conversions=int(clicks*bm['cvr']*np.random.uniform(0.80,1.20))
            brand_lift=round(np.random.uniform(0.015,0.055)*sm*decay,4)
            view_rate=round(np.random.uniform(0.30,0.65),3)

        # Anomaly impacts
        if wi in [18,19] and market=='India':
            brand_lift=round(brand_lift*0.35,4); conversions=int(conversions*0.50)
        if wi in [55,56] and 'World Cup' in name:
            brand_lift=round(brand_lift*2.2,4); impressions=int(impressions*1.8)
        if wi in [68,69] and market=='US':
            brand_lift=round(brand_lift*0.50,4); conversions=int(conversions*0.60)

        rows.append({
            'week':               week_labels[wi],
            'week_start_date':    week_starts[wi].strftime('%Y-%m-%d'),
            'market':             market,
            'campaign_name':      name,
            'campaign_type':      ctype,
            'flavor':             flavor,           # ← flavor for ALL campaigns
            'week_in_campaign':   wic+1,
            'impressions':        impressions,
            'reach':              reach,
            'clicks':             clicks,
            'ctr':                ctr,
            'conversions':        conversions,
            'view_completion_rate': view_rate,
            'spend_inr':          spend_inr if market=='India' else 0,
            'spend_usd':          int(spend_inr*0.012) if market=='US' else 0,
            'brand_lift':         brand_lift,
            'roas':               round(conversions*280/spend_inr,3) if spend_inr>0 and conversions>0 else 0,
            'is_anomaly_week':    wi in ANOMALIES,
            'anomaly_type':       ANOMALIES.get(wi,{}).get('type',''),
        })

campaign_df = pd.DataFrame(rows)
campaign_df.to_csv(f'{OUTPUT_DIR}/campaign_metrics.csv', index=False)
print(f'✅ campaign_metrics.csv → {len(campaign_df):,} rows')
print(f'   Flavors in campaigns: {sorted(campaign_df["flavor"].unique())}')
# ─── Cell 6: Dataset 4 — Product Reviews (All brands) ────────────────────────

PLATFORMS_REVIEW = {'India':['Amazon India','Flipkart'], 'US':['Amazon US','Walmart']}

REVIEW_TEMPLATES = {
    "Lay's": {
        5: {'India': [
                "Ekdum mast hai Lay's {flavor}! Fresh aur crunchy. Family favorite 😄",
                "Superb quality! Lay's {flavor} consistent rehta hai. Highly recommend!",
                "Lay's {flavor} is literally the best chips in India. Paise vasool!",
            ],
            'US': [
                "Lay's {flavor} is consistently the best chip out there. Never disappoints.",
                "Bought for the Super Bowl party — gone in 20 mins. Lay's {flavor} always wins!",
                "My go-to for years. Lay's {flavor} always fresh, always delicious.",
            ]},
        4: {'India': [
                "Lay's {flavor} achha hai but quantity thodi kam hai packet mein. Taste great though.",
                "Good product. Lay's {flavor} fresh tha. Packaging thoda improve ho sakta hai.",
            ],
            'US': [
                "Good as always. Lay's {flavor} consistent quality. Wish bags were fuller.",
                "Really good chip. Lay's {flavor} delivers on flavor. A few got crushed in shipping.",
            ]},
        3: {'India': [
                "Theek hai Lay's {flavor}. Expected better from such a big brand.",
                "Average. Lay's {flavor} pehle zyada achha lagta tha.",
            ],
            'US': [
                "Lay's {flavor} is decent but I've had better. Nothing special.",
                "Average chip. Lay's {flavor} used to taste better.",
            ]},
        2: {'India': [
                "Bahut disappointed. Lay's {flavor} stale aaya. Expiry 3 months baad hai phir bhi.",
                "Packet mein chips tootey hue the. Lay's {flavor} packaging weak hai.",
            ],
            'US': [
                "Lay's {flavor} arrived stale despite expiry being months away. Disappointed.",
                "Half the bag was crushed. Lay's {flavor} packaging terrible for shipping.",
            ]},
        1: {'India': [
                "DISGUSTING! Lay's {flavor} mein kuch aisa mila jo chips nahi tha. @Lays jawab do!!",
                "Bahut bura experience. Lay's {flavor} tampered packet. Consumer forum mein complaint.",
                "Zero quality control. Lay's {flavor} stale. Smell bhi ajeeb tha. AVOID!",
            ],
            'US': [
                "Found something disturbing in my Lay's {flavor} bag. Filing FDA complaint.",
                "Worst experience. Lay's {flavor} was completely rancid. Threw it out.",
                "ZERO stars. Lay's {flavor} had something that was NOT a chip. Disgusting.",
            ]},
    },
    'Pringles': {
        5: {'India': ["Pringles {flavor} ekdum fresh! Canister packaging is the best 👌",
                      "Pringles {flavor} — consistent quality every single time. Love it!"],
            'US':    ["Pringles {flavor} never fails. Perfect crunch, great flavor.",
                      "Once you pop you can't stop — {flavor} is my all-time fave Pringles!"]},
        4: {'India': ["Pringles {flavor} good hai. Thoda expensive but worth it."],
            'US':    ["Good chip. Pringles {flavor} has solid flavor. Canister makes it easy to share."]},
        3: {'India': ["Pringles {flavor} theek hai. Lay's se thoda zyada price ke liye expected better."],
            'US':    ["Pringles {flavor} is fine. Nothing special at this price point."]},
        2: {'India': ["Pringles {flavor} stale aaya. Disappointed with freshness."],
            'US':    ["Pringles {flavor} didn't taste fresh. Expected better shelf management."]},
        1: {'India': ["Pringles {flavor} mein kuch ajeeb smell tha. Waste of money."],
            'US':    ["Pringles {flavor} was completely stale. Returning this."]},
    },
    'Doritos': {
        5: {'India': ["Doritos {flavor} is fire! Perfect for gaming nights 🎮🔥",
                      "Doritos {flavor} ka taste India mein bhi utna hi amazing hai!"],
            'US':    ["Doritos {flavor} is the perfect snack. Bold flavor, great crunch.",
                      "Doritos {flavor} with salsa is a religious experience 🌮"]},
        4: {'India': ["Doritos {flavor} good hai. Thoda zyada spicy but I liked it."],
            'US':    ["Doritos {flavor} solid as always. Maybe slightly too salty but still great."]},
        3: {'India': ["Doritos {flavor} okay hai. Pricey for what you get."],
            'US':    ["Doritos {flavor} is fine. Not my fave but decent enough."]},
        2: {'India': ["Doritos {flavor} stale tha. Brand trust thoda gira hai recent controversy ke baad."],
            'US':    ["Doritos {flavor} stale. Also the influencer controversy really put me off."]},
        1: {'India': ["Doritos {flavor} waste hai. Influencer controversy ke baad trust nahi raha."],
            'US':    ["Won't buy Doritos {flavor} again after that influencer scandal. Done."]},
    }
}

BASE_REVIEWS = {'India': 38, 'US': 42}
BRAND_REVIEW_SHARE = {"Lay's": 0.55, 'Pringles': 0.25, 'Doritos': 0.20}

rows = []
review_id = 1

for w_idx, (w_start, w_label) in enumerate(zip(week_starts, week_labels)):
    for market in MARKETS:
        sm = seasonal_multiplier(w_idx, market)
        _, vol_mult = anomaly_sentiment_delta(w_idx, market)
        total_count = int(BASE_REVIEWS[market] * sm * vol_mult * np.random.uniform(0.80,1.20))

        for brand, share in BRAND_REVIEW_SHARE.items():
            count = clamp(int(total_count * share), 2, 400)

            # Rating weights
            if brand=="Lay's" and w_idx==18 and market=='India':   weights=[0.35,0.22,0.15,0.15,0.13]
            elif brand=="Lay's" and w_idx==19 and market=='India': weights=[0.22,0.18,0.18,0.22,0.20]
            elif brand=="Lay's" and w_idx in [55,56]:              weights=[0.03,0.05,0.10,0.28,0.54]
            elif brand=="Lay's" and w_idx in [68,69] and market=='US': weights=[0.20,0.20,0.18,0.22,0.20]
            elif brand=='Doritos' and w_idx in [42,43]:            weights=[0.18,0.20,0.20,0.22,0.20]
            elif brand=='Pringles' and w_idx in range(28,36):      weights=[0.03,0.06,0.12,0.32,0.47]
            else:                                                   weights=[0.05,0.08,0.13,0.30,0.44]

            for _ in range(count):
                platform   = random.choice(PLATFORMS_REVIEW[market])
                flavor     = get_flavor(brand, w_idx)
                rating     = random.choices([1,2,3,4,5], weights=weights)[0]
                tmpl_pool  = REVIEW_TEMPLATES[brand][rating][market]
                text       = random.choice(tmpl_pool).format(flavor=flavor)
                review_date= w_start + timedelta(days=random.randint(0,6))

                rows.append({
                    'review_id':        f'R{review_id:07d}',
                    'week':             w_label,
                    'week_start_date':  w_start.strftime('%Y-%m-%d'),
                    'review_date':      review_date.strftime('%Y-%m-%d'),
                    'market':           market,
                    'platform':         platform,
                    'brand':            brand,
                    'flavor':           flavor,          # ← flavor for ALL brands
                    'star_rating':      rating,
                    'review_text':      text,
                    'helpful_votes':    int(np.random.lognormal(1.8,1.2)),
                    'verified_purchase': random.choices([True,False], weights=[0.82,0.18])[0],
                    'is_anomaly_week':  w_idx in ANOMALIES,
                    'anomaly_type':     ANOMALIES.get(w_idx,{}).get('type',''),
                })
                review_id += 1

reviews_df = pd.DataFrame(rows)
reviews_df.to_csv(f'{OUTPUT_DIR}/reviews.csv', index=False)
print(f'✅ reviews.csv → {len(reviews_df):,} rows')
for b in BRANDS:
    flavors = sorted(reviews_df[reviews_df['brand']==b]['flavor'].unique())
    print(f'   {b} flavors: {flavors}')
# ─── Cell 7: Dataset 5 — Competitor News ─────────────────────────────────────

COMPETITOR_NEWS = [
    (1,  'Pringles','US',    'Pringles kicks off 2025 with Super Bowl party pack bundle — targets Lay\'s game-day occasion','promotion',-0.03,'AdWeek','Original'),
    (3,  'Doritos', 'India', 'Doritos expands to 800 new Tier-2 city outlets, doubling Nacho Cheese shelf presence','distribution',-0.03,'Economic Times','Nacho Cheese'),
    (5,  'Pringles','US',    'Pringles signs NFL deal — Sour Cream & Onion ads across all 2025 playoff games','partnership',-0.04,'Sports Business Journal','Sour Cream & Onion'),
    (8,  'Doritos', 'India', 'WHO-linked report on ultra-processed snacks — Nacho Cheese and similar flavors called out','external',-0.06,'Mint','Nacho Cheese'),
    (8,  'Pringles','US',    'Health groups target chip brands — Pringles Original and Lay\'s Classic both face negative press','external',-0.05,'Business Insider','Original'),
    (12, 'Pringles','India', 'Pringles India signs celebrity ambassador — BBQ flavor featured in launch campaign','marketing',-0.04,'Mint','BBQ'),
    (18, 'Pringles','India', 'Pringles India sales jump 22% as consumers switch from Lay\'s amid packaging controversy','competitive',-0.06,'Business Standard','Original'),
    (19, 'Pringles','India', 'Retailers report Pringles Original demand up 35% WoW — metros running low on stock','competitive',-0.05,'Economic Times','Original'),
    (22, 'Doritos', 'US',    'Doritos partners with NBA for March Madness — Cool Ranch 60-second spots on ESPN','marketing',-0.03,'AdAge','Cool Ranch'),
    (26, 'Pringles','US',    'Pringles Q1 2025: North America revenue up 11% — Sour Cream & Onion leads growth','financial',-0.03,'Reuters','Sour Cream & Onion'),
    (28, 'Pringles','India', 'Pringles launches Wavy Classic and Wavy Ranch in India — priced to compete with Lay\'s ₹20 pack','product',-0.05,'Economic Times','Wavy Classic'),
    (28, 'Pringles','US',    'Pringles Wavy hits US shelves — Walmart gives endcap display replacing Lay\'s slot','distribution',-0.05,'Grocery Dive','Wavy Ranch'),
    (30, 'Pringles','India', 'Pringles India announces 20% price cut across ALL SKUs including Wavy range — effective immediately','pricing',-0.07,'Business Standard','Wavy Classic'),
    (30, 'Pringles','US',    'Pringles drops 18% in the US — analysts say Wavy targets Lay\'s price-sensitive base','pricing',-0.07,'Wall Street Journal','Wavy Classic'),
    (31, 'Pringles','US',    'Week 2 of Pringles price cut: volume up 41%, Lay\'s market share dips 2.3pts in Nielsen data','pricing',-0.06,'Nielsen via AdAge','Original'),
    (36, 'Doritos', 'India', 'Doritos Flamin Hot goes viral on Reels — 8M views in 48 hours for Diwali campaign','marketing',-0.04,'Social Samosa','Flamin Hot'),
    (38, 'Pringles','India', 'Pringles Diwali gift tin featuring BBQ and Original sells out on Amazon India in 36 hours','promotion',-0.04,'Economic Times','BBQ'),
    (40, 'Doritos', 'US',    'Doritos Flamin Hot Lime teased on social ahead of launch — massive pre-launch buzz','product',-0.02,'Marketing Brew','Flamin Hot Lime'),
    (42, 'Doritos', 'US',    'Doritos influencer scandal: creator uses Cool Ranch campaign to make controversial statement — brand apologises','controversy',+0.03,'AdWeek','Cool Ranch'),
    (42, 'Doritos', 'India', 'Doritos Cool Ranch influencer controversy trends on Indian Twitter — #BoycottDoritos 180K tweets','controversy',+0.03,'Social Samosa','Cool Ranch'),
    (43, 'Doritos', 'US',    'Doritos brand trust score drops 14pts in YouGov — Cool Ranch sales down 18% WoW','controversy',+0.04,'YouGov via Marketing Week','Cool Ranch'),
    (48, 'Doritos', 'US',    'Doritos Flamin Hot Lime officially launches in US — strong early reviews despite brand still recovering','product',-0.02,'Snack Food & Wholesale Bakery','Flamin Hot Lime'),
    (48, 'Doritos', 'India', 'Doritos Flamin Hot Lime arrives in India — positioned as premium Rs 30 pack','product',-0.02,'Economic Times','Flamin Hot Lime'),
    (55, 'Pringles','India', 'Pringles tries to ride cricket wave with last-minute Original push — response lukewarm vs Lay\'s WC launch','marketing',+0.02,'Mint','Original'),
    (60, 'Pringles','US',    'Pringles H1 2025: market share up 1.8pts — Wavy range credited for gains against Lay\'s','financial',-0.03,'Reuters','Wavy Classic'),
    (65, 'Doritos', 'US',    'Doritos Nacho Cheese relaunch campaign — brand attempting comeback after influencer scandal','marketing',-0.02,'AdAge','Nacho Cheese'),
    (68, 'Pringles','US',    'Pringles Original fills shelf gaps as Lay\'s faces US supply chain disruption','competitive',-0.05,'Grocery Dive','Original'),
    (69, 'Doritos', 'US',    'Doritos Spicy Nacho benefits from Lay\'s US stock shortage — convenience stores report surge','competitive',-0.04,'Path to Purchase','Spicy Nacho'),
    (72, 'Pringles','India', 'Pringles Cheddar Cheese variant launches India — positioned as premium snack segment entry','product',-0.03,'Business Standard','Cheddar Cheese'),
]

rows = []
for item in COMPETITOR_NEWS:
    w_idx,brand,market,headline,category,lays_impact,source,flavor = item
    if w_idx >= WEEKS: continue
    news_date = week_starts[w_idx] + timedelta(days=random.randint(0,5))
    rows.append({
        'week':                           week_labels[w_idx],
        'week_start_date':                week_starts[w_idx].strftime('%Y-%m-%d'),
        'news_date':                      news_date.strftime('%Y-%m-%d'),
        'market':                         market,
        'competitor_brand':               brand,
        'flavor':                         flavor,          # ← flavor in competitor news
        'headline':                       headline,
        'category':                       category,
        'estimated_lays_sentiment_impact':lays_impact,
        'source':                         source,
        'is_anomaly_week':                w_idx in ANOMALIES,
        'anomaly_type':                   ANOMALIES.get(w_idx,{}).get('type',''),
    })

competitor_df = pd.DataFrame(rows)
competitor_df.to_csv(f'{OUTPUT_DIR}/competitor_news.csv', index=False)
print(f'✅ competitor_news.csv → {len(competitor_df):,} rows')
print(f'   Flavors tracked: {sorted(competitor_df["flavor"].unique())}')
# ─── Cell 8: Dataset 6 — Brand Tracker Summary ───────────────────────────────

rows = []
for w_idx, (w_start, w_label) in enumerate(zip(week_starts, week_labels)):
    for market in MARKETS:
        sm = seasonal_multiplier(w_idx, market)
        sent_delta, _ = anomaly_sentiment_delta(w_idx, market)

        awareness       = clamp(0.74+sm*0.04 +np.random.normal(0,0.012)+sent_delta*0.25, 0.40,0.95)
        consideration   = clamp(0.56+sm*0.035+np.random.normal(0,0.010)+sent_delta*0.35, 0.30,0.88)
        purchase_intent = clamp(0.41+sm*0.05 +np.random.normal(0,0.011)+sent_delta*0.45, 0.18,0.78)
        nps             = clamp(44  +sm*4    +np.random.normal(0,2.5)  +sent_delta*55,   -30, 85)
        brand_sentiment = clamp(0.64+sent_delta+np.random.normal(0,0.018),                0.08,0.92)

        lays_sov    = clamp(0.46+sm*0.02+np.random.normal(0,0.012)+sent_delta*0.18+(0.08 if w_idx in [55,56,57] else 0), 0.18,0.72)
        pringles_sov= clamp(0.29-sent_delta*0.12+(0.07 if w_idx in range(28,36) else 0)+np.random.normal(0,0.011),       0.12,0.55)
        doritos_sov = clamp(0.25+(0.04 if w_idx in [42,43] else 0)+np.random.normal(0,0.010),                            0.08,0.48)

        rows.append({
            'week':                    w_label,
            'week_start_date':         w_start.strftime('%Y-%m-%d'),
            'market':                  market,
            'brand':                   "Lay's",
            'awareness':               round(awareness,4),
            'consideration':           round(consideration,4),
            'purchase_intent':         round(purchase_intent,4),
            'net_promoter_score':      round(nps,1),
            'brand_sentiment_score':   round(brand_sentiment,4),
            'share_of_voice_lays':     round(lays_sov,4),
            'share_of_voice_pringles': round(pringles_sov,4),
            'share_of_voice_doritos':  round(doritos_sov,4),
            'category_sentiment':      round(clamp(0.55+sent_delta*0.4+np.random.normal(0,0.015),0.20,0.85),4),
            'is_anomaly_week':         w_idx in ANOMALIES,
            'anomaly_type':            ANOMALIES.get(w_idx,{}).get('type',''),
            'anomaly_description':     ANOMALIES.get(w_idx,{}).get('desc',''),
        })

tracker_df = pd.DataFrame(rows)
tracker_df.to_csv(f'{OUTPUT_DIR}/brand_tracker_summary.csv', index=False)
print(f'✅ brand_tracker_summary.csv → {len(tracker_df):,} rows')
# ─── Cell 9: Dataset 7 — Weekly KPI Dashboard ────────────────────────────────

rows = []
for w_idx, (w_start, w_label) in enumerate(zip(week_starts, week_labels)):
    for market in MARKETS:
        soc  = social_df[(social_df['week']==w_label)&(social_df['market']==market)&(social_df['brand_mentioned']=="Lay's")]
        srch = search_df[(search_df['week']==w_label)&(search_df['market']==market)&(search_df['brand']=="Lay's")]
        rev  = reviews_df[(reviews_df['week']==w_label)&(reviews_df['market']==market)&(reviews_df['brand']=="Lay's")]
        camp = campaign_df[(campaign_df['week']==w_label)&(campaign_df['market']==market)]
        trk  = tracker_df[(tracker_df['week']==w_label)&(tracker_df['market']==market)]
        comp = competitor_df[(competitor_df['week']==w_label)&(competitor_df['market']==market)]

        total_posts = len(soc)
        pos_pct     = round(len(soc[soc['sentiment']=='positive'])/max(total_posts,1),4)
        neg_pct     = round(len(soc[soc['sentiment']=='negative'])/max(total_posts,1),4)
        total_eng   = int(soc['total_engagement'].sum())

        # Top flavor by engagement this week
        if total_posts > 0:
            top_flavor = soc.groupby('flavor')['total_engagement'].sum().idxmax()
        else:
            top_flavor = ''

        avg_rating    = round(rev['star_rating'].mean(),2) if len(rev)>0 else np.nan
        review_count  = len(rev)
        pct_1star     = round(len(rev[rev['star_rating']==1])/max(review_count,1),4)

        awareness  = float(trk['awareness'].values[0])           if len(trk)>0 else np.nan
        sov        = float(trk['share_of_voice_lays'].values[0]) if len(trk)>0 else np.nan
        nps        = float(trk['net_promoter_score'].values[0])  if len(trk)>0 else np.nan
        brand_sent = float(trk['brand_sentiment_score'].values[0]) if len(trk)>0 else np.nan

        health_score = round(
            (pos_pct*25) +
            (min(avg_rating/5,1)*25 if not np.isnan(avg_rating) else 12.5) +
            (min(sov,1)*25 if not np.isnan(sov) else 12.5) +
            (min(max(nps+30,0)/110,1)*25 if not np.isnan(nps) else 12.5), 2
        )

        rows.append({
            'week':                   w_label,
            'week_start_date':        w_start.strftime('%Y-%m-%d'),
            'market':                 market,
            'total_social_posts':     total_posts,
            'positive_sentiment_pct': pos_pct,
            'negative_sentiment_pct': neg_pct,
            'total_engagement':       total_eng,
            'top_flavor_by_engagement': top_flavor,    # ← flavor in KPI dashboard
            'total_search_volume':    int(srch['search_volume'].sum()),
            'avg_review_rating':      avg_rating,
            'total_reviews':          review_count,
            'pct_1star_reviews':      pct_1star,
            'campaign_impressions':   int(camp['impressions'].sum()),
            'campaign_brand_lift':    round(camp['brand_lift'].mean(),4) if len(camp)>0 else 0.0,
            'active_campaigns':       len(camp['campaign_name'].unique()),
            'spend_inr':              int(camp['spend_inr'].sum()),
            'spend_usd':              int(camp['spend_usd'].sum()),
            'brand_awareness':        round(awareness,4),
            'share_of_voice':         round(sov,4),
            'net_promoter_score':     round(nps,1),
            'brand_sentiment_score':  round(brand_sent,4),
            'competitor_news_count':  len(comp),
            'competitor_threat_score':round(abs(comp['estimated_lays_sentiment_impact'].sum()),3),
            'brand_health_score':     health_score,
            'is_anomaly_week':        w_idx in ANOMALIES,
            'anomaly_type':           ANOMALIES.get(w_idx,{}).get('type',''),
            'anomaly_description':    ANOMALIES.get(w_idx,{}).get('desc',''),
        })

kpi_df = pd.DataFrame(rows)
kpi_df.to_csv(f'{OUTPUT_DIR}/weekly_kpi_dashboard.csv', index=False)
print(f'✅ weekly_kpi_dashboard.csv → {len(kpi_df):,} rows')
# ─── Cell 10: Final Validation ────────────────────────────────────────────────
import os

files = [
    ('social_posts.csv',          social_df),
    ('search_trends.csv',         search_df),
    ('campaign_metrics.csv',      campaign_df),
    ('reviews.csv',               reviews_df),
    ('competitor_news.csv',       competitor_df),
    ('brand_tracker_summary.csv', tracker_df),
    ('weekly_kpi_dashboard.csv',  kpi_df),
]

print('='*70)
print("📦  LAY'S BRAND HEALTH — SYNTHETIC DATA v3 COMPLETE")
print('='*70)
for fname, df in files:
    size_kb = os.path.getsize(f'{OUTPUT_DIR}/{fname}')/1024
    has_flavor = '✅ flavor col' if 'flavor' in df.columns else '❌ no flavor'
    print(f'  {fname:<38} {len(df):>7,} rows  {size_kb:>8.1f} KB  {has_flavor}')

print()
print(f'📅 Period  : {START_DATE.strftime("%b %Y")} → {TODAY.strftime("%b %Y")} (78 weeks / 18 months)')
print('🌏 Markets : India + US')
print('🏷  Language: English + Hinglish (India social posts)')
print()
print('🍟 Flavors tracked across ALL datasets:')
for brand in BRANDS:
    print(f'   {brand:<12}: {FLAVORS[brand]}')
print()
print('🔴 6 Anomaly events:')
for w, info in ANOMALIES.items():
    print(f'   Week {w:>2} ({info["type"]:<18}) → {info["desc"]}')
print()
print('✅ Ready for Databricks Delta tables + Streamlit agent pipeline!')