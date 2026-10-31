#!/usr/bin/env python3
"""Build reproducible English labels for Citywalk network and source-place names."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from pypinyin import lazy_pinyin


FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parent
NODES_CSV = ROOT / "data" / "analysis" / "spatial_figure_nodes.csv"
RECORDS_CSV = ROOT / "results" / "subnetworks" / "theme_subnetwork_records.csv"
OUTPUT_CSV = FRONTEND / "place_name_translations.csv"

CHINESE = re.compile(r"[\u3400-\u9fff]")
CHINESE_RUN = re.compile(r"[\u3400-\u9fff]+")

EXACT_TRANSLATIONS = {
    "皓月湖": "Haoyue Lake",
    "溪头站": "Xitou Station",
    "角美菜市场": "Jiaomei Produce Market",
    "天竺山": "Tianzhu Mountain",
    "仙灵棋山": "Xianlingqi Mountain",
    "西堤咖啡别墅一条街": "Xidi Coffee Villa Street",
    "塔头公交站": "Tatou Bus Stop",
    "下沉广场": "Sunken Plaza",
    "莲花社区": "Lianhua Community",
    "华新路": "Huaxin Road",
    "幸福路": "Xingfu Road",
    "黄厝沙滩": "Huangcuo Beach",
    "黄厝海滩": "Huangcuo Beach",
    "黄厝海滨": "Huangcuo Seafront",
    "溪岸路花鸟市场": "Xi'an Road Flower and Bird Market",
    "溪岸花鸟市场": "Xi'an Road Flower and Bird Market",
    "花鸟市场": "Flower and Bird Market",
    "八市": "Eighth Market",
    "第八市场": "Eighth Market",
    "中山路": "Zhongshan Road",
    "中山路步行街": "Zhongshan Road Pedestrian Street",
    "中山公园": "Zhongshan Park",
    "南普陀寺": "Nanputuo Temple",
    "集美学村": "Jimei School Village",
    "十里长堤": "Shili Long Causeway",
    "百家村": "Baijia Village",
    "陈嘉庚故居": "Former Residence of Tan Kah Kee",
    "陈嘉庚纪念馆": "Tan Kah Kee Memorial Museum",
    "嘉庚公园": "Kah Kee Park",
    "鳌园": "Ao Garden",
    "龙舟池": "Dragon Boat Pond",
    "集美大社": "Jimei Dashe",
    "鼓浪屿": "Gulangyu Island",
    "鼓浪屿岛": "Gulangyu Island",
    "厦门大学": "Xiamen University",
    "厦大": "Xiamen University",
    "沙坡尾": "Shapowei",
    "沙坡尾艺术西区": "Shapowei Art Zone",
    "白鹭洲公园": "Bailuzhou Park",
    "海湾公园": "Haiwan Park",
    "环岛路": "Island Ring Road",
    "环岛南路": "South Island Ring Road",
    "曾厝垵": "Zengcuo'an",
    "曾厝垵村": "Zengcuo'an Village",
    "铁路文化公园": "Railway Culture Park",
    "五缘湾": "Wuyuan Bay",
    "五缘湾湿地公园": "Wuyuan Bay Wetland Park",
    "筼筜湖": "Yundang Lake",
    "员筜湖": "Yundang Lake",
    "万石植物园": "Wanshi Botanical Garden",
    "厦门园林植物园": "Xiamen Botanical Garden",
    "厦门植物园": "Xiamen Botanical Garden",
    "双子塔": "Xiamen Twin Towers",
    "世茂双子塔": "Shimao Twin Towers",
    "演武大桥": "Yanwu Bridge",
    "胡里山炮台": "Hulishan Fortress",
    "顶澳仔猫街": "Ding'aozai Cat Street",
    "猫街": "Cat Street",
    "钟鼓索道": "Bell and Drum Cableway",
    "厦门北站": "Xiamen North Railway Station",
    "厦门站": "Xiamen Railway Station",
    "厦门火车站": "Xiamen Railway Station",
    "高崎机场": "Xiamen Gaoqi International Airport",
    "厦门高崎国际机场": "Xiamen Gaoqi International Airport",
    "邮轮中心厦鼓码头": "Cruise Center Xiamen-Gulangyu Ferry Terminal",
    "厦鼓码头": "Xiamen-Gulangyu Ferry Terminal",
    "轮渡码头": "Ferry Terminal",
    "第一码头": "First Pier",
    "和平码头": "Heping Wharf",
    "海沧大桥": "Haicang Bridge",
    "厦门大桥": "Xiamen Bridge",
    "集美大桥": "Jimei Bridge",
    "杏林大桥": "Xinglin Bridge",
    "海上世界": "Sea World",
    "观音山": "Guanyin Mountain",
    "观音山沙滩": "Guanyin Mountain Beach",
    "椰风寨": "Yefengzhai",
    "珍珠湾": "Pearl Bay",
    "白城沙滩": "Baicheng Beach",
    "一国两制沙滩": "One Country, Two Systems Beach",
    "厦门科技馆": "Xiamen Science and Technology Museum",
    "厦门市博物馆": "Xiamen Museum",
    "厦门博物馆": "Xiamen Museum",
    "厦门美术馆": "Xiamen Art Museum",
    "文化艺术中心": "Xiamen Culture and Arts Center",
    "厦门图书馆": "Xiamen Library",
    "厦门市图书馆": "Xiamen Library",
    "厦门体育中心": "Xiamen Sports Center",
    "厦门国际会展中心": "Xiamen International Conference and Exhibition Center",
    "南湖公园": "Nanhu Park",
    "鸿山公园": "Hongshan Park",
    "仙岳山": "Xianyue Mountain",
    "仙岳公园": "Xianyue Park",
    "狐尾山公园": "Huweishan Park",
    "东坪山": "Dongpingshan",
    "东坪山公园": "Dongpingshan Park",
    "梅海岭": "Meihailing",
    "梅海花台": "Meihai Flower Terrace",
    "上李水库": "Shangli Reservoir",
    "云顶山": "Yunding Mountain",
    "天泉步道广场": "Tianquan Trail Plaza",
    "山海健康步道": "Mountain-Sea Health Trail",
    "厦门山海健康步道": "Xiamen Mountain-Sea Health Trail",
    "山海步道": "Mountain-Sea Trail",
    "海上明珠塔": "Pearl Tower",
    "园博苑": "Xiamen Horticulture Expo Garden",
    "厦门园博苑": "Xiamen Horticulture Expo Garden",
    "园博园": "Horticulture Expo Garden",
    "集美新城": "Jimei New Town",
    "海沧湾公园": "Haicang Bay Park",
    "海沧湖公园": "Haicang Lake Park",
    "大屏山郊野公园": "Dapingshan Country Park",
    "天语舟气象公园": "Tianyuzhou Meteorological Park",
    "同安影视城": "Tong'an Film Studio",
    "方特梦幻王国": "Fantawild Dreamland",
    "方特东方神画": "Fantawild Oriental Heritage",
    "方特水上乐园": "Fantawild Water Park",
    "北辰山": "Beichen Mountain",
    "金光湖": "Jinguang Lake",
    "梵天寺": "Fantian Temple",
    "苏颂公园": "Su Song Park",
    "翔安香山": "Xiang'an Xiangshan",
    "大嶝岛": "Dadeng Island",
    "小嶝岛": "Xiaodeng Island",
    "下潭尾红树林公园": "Xiatanwei Mangrove Park",
    "厦门大学翔安校区": "Xiamen University Xiang'an Campus",
    "张万森家": "Zhang Wansen's House",
    "张万森的家": "Zhang Wansen's House",
    "张万森家门口的炮仗花": "Bougainvillea at Zhang Wansen's House",
    "林北星家": "Lin Beixing's House",
    "林北星的家": "Lin Beixing's House",
    "不在书店": "On the Road Bookstore",
    "做个闲人": "Zuoge Xianren",
    "良筑": "Liangzhu",
    "音乐广场": "Music Square",
    "斗西路": "Douxi Road",
    "日光岩": "Sunlight Rock",
    "盼归塔": "Pangui Tower",
    "石鼓路": "Shigu Road",
    "菽庄花园": "Shuzhuang Garden",
    "公园西路": "Gongyuan West Road",
    "公园东路": "Gongyuan East Road",
    "公园南路": "Gongyuan South Road",
    "溪头下": "Xitouxia",
    "溪岸路": "Xi'an Road",
    "园林植物园": "Xiamen Botanical Garden",
    "皓月园": "Haoyue Garden",
    "宇宙有趣研究所": "Universe Fun Institute",
    "最美转角": "Most Beautiful Corner",
    "三丘田码头": "Sanqiutian Ferry Terminal",
    "海韵台": "Haiyuntai",
    "大中沙茶面": "Dazhong Satay Noodles",
    "蛀牙糖水": "Zhuya Dessert Shop",
    "内厝澳码头": "Neicuo'ao Ferry Terminal",
    "南堤公园": "Nandi Park",
    "中华城": "China City Mall",
    "风琴博物馆": "Organ Museum",
    "钢琴码头": "Piano Pier",
    "姜母鸭": "Ginger Duck",
    "沙茶面": "Satay Noodles",
    "楚雨荨的家": "Chu Yuxun's House",
    "信义里": "Xinyi Lane",
    "山海步道林海线": "Mountain-Sea Trail: Linhai Route",
    "山海健康步道林海线": "Mountain-Sea Health Trail: Linhai Route",
    "大社戏台": "Dashe Theatre Stage",
    "海上自行车道": "Coastal Cycleway",
    "管风琴艺术中心": "Organ Art Center",
    "南薰楼群": "Nanxun Building Complex",
    "草埔尾路": "Caopuwei Road",
    "圆月广场": "Full Moon Plaza",
    "新华路": "Xinhua Road",
    "海韵台站": "Haiyuntai Station",
    "海堤纪念公园": "Seawall Memorial Park",
    "莲花公园": "Lianhua Park",
    "八卦楼": "Bagua Mansion",
    "四果汤": "Four-Fruit Sweet Soup",
    "民族路": "Minzu Road",
    "港仔后沙滩": "Gangzaihou Beach",
    "兴华路": "Xinghua Road",
    "野路甜品铺": "Yelu Dessert Shop",
    "野路甜点铺": "Yelu Dessert Shop",
    "玄妙巷": "Xuanmiao Lane",
    "美仁宫农贸市场": "Meirengong Farmers' Market",
    "天主堂": "Catholic Church",
    "天主教堂": "Catholic Church",
    "花生汤": "Peanut Soup",
    "溪头下站": "Xitouxia Station",
    "潮汐之眼摩天轮": "Eye of the Tide Ferris Wheel",
    "三官道院": "Sanguan Taoist Temple",
    "厦门墙": "Xiamen Wall",
    "会展中心": "Conference and Exhibition Center",
    "怡海小镇": "Yihai Town",
    "中兴路": "Zhongxing Road",
    "晴天墙": "Sunny Wall",
    "文庙": "Confucian Temple",
    "演武大桥观景台": "Yanwu Bridge Viewpoint",
    "演武大桥观景平台": "Yanwu Bridge Viewing Platform",
    "后江埭路": "Houjiangdai Road",
    "花姐海鲜加工": "Huajie Seafood Kitchen",
    "向日葵之家": "Sunflower House",
    "书法广场": "Calligraphy Square",
    "天后宫": "Tianhou Temple",
    "西堤咖啡一条街": "Xidi Coffee Street",
    "彩虹沙滩": "Rainbow Beach",
    "月光岩": "Moonlight Rock",
    "局口拌面": "Jukou Noodles",
    "五缘湾帆船": "Wuyuan Bay Sailing",
    "五缘湾帆船港": "Wuyuan Bay Sailing Marina",
    "北门外街": "Beimenwai Street",
    "日光岩顶端": "Summit of Sunlight Rock",
    "鹭江夜游": "Lujiang Night Cruise",
    "龙头路": "Longtou Road",
    "金沙书院": "Jinsha Academy",
    "毓园": "Yuyuan Garden",
    "芙蓉隧道": "Furong Tunnel",
    "老院子": "Old Courtyard",
    "万象城": "The MixC",
    "建业路": "Jianye Road",
    "云水谣古镇": "Yunshuiyao Ancient Town",
    "珍珠湾步道": "Pearl Bay Trail",
    "华新路33号": "No. 33 Huaxin Road",
    "厦大附近": "Near Xiamen University",
    "斗西便利店": "Douxi Convenience Store",
    "大同路": "Datong Road",
    "海底世界": "Underwater World",
    "同安钟楼": "Tong'an Clock Tower",
    "石鼓路小吃街": "Shigu Road Snack Street",
    "思北": "Sibei",
    "水秀广场": "Shuixiu Square",
    "尚南路": "Shangnan Road",
    "斗西路口": "Douxi Road Junction",
    "中山街": "Zhongshan Street",
    "望湖塔": "Wanghu Tower",
    "松柏公园": "Songbai Park",
    "曾厝垵音乐广场": "Zengcuo'an Music Square",
    "音乐广场沙滩": "Music Square Beach",
    "美仁广场": "Meiren Square",
    "图强路": "Tuqiang Road",
    "海天堂构": "Hi Heaven Villa",
    "大德记沙滩": "Dadeji Beach",
    "思北花生汤": "Sibei Peanut Soup",
    "叶氏麻糍": "Ye's Mochi",
    "五通灯塔公园": "Wutong Lighthouse Park",
    "彩虹阶梯": "Rainbow Stairs",
    "福音堂": "Gospel Church",
    "鹭江道": "Lujiang Road",
    "大屏山公园": "Dapingshan Park",
    "滨海浪漫线": "Coastal Romantic Route",
    "金榜公园": "Jinbang Park",
    "虎头山公园": "Hutoushan Park",
    "溪头下沙滩": "Xitouxia Beach",
    "金沙湾": "Jinsha Bay",
    "嘉庚故居": "Former Residence of Tan Kah Kee",
    "植物园西门": "West Gate of Xiamen Botanical Garden",
    "禾祥西路": "Hexiang West Road",
    "西堤": "Xidi",
    "嘉庚路": "Kah Kee Road",
    "世贸双子塔": "Shimao Twin Towers",
    "华侨博物院": "Overseas Chinese Museum",
}

PHRASE_TRANSLATIONS = {
    "厦门": "Xiamen",
    "思明": "Siming",
    "湖里": "Huli",
    "集美": "Jimei",
    "海沧": "Haicang",
    "同安": "Tong'an",
    "翔安": "Xiang'an",
    "鼓浪屿": "Gulangyu",
    "中山路": "Zhongshan Road",
    "中山公园": "Zhongshan Park",
    "华新路": "Huaxin Road",
    "幸福路": "Xingfu Road",
    "南普陀寺": "Nanputuo Temple",
    "南普陀": "Nanputuo",
    "厦门大学": "Xiamen University",
    "厦大": "Xiamen University",
    "集美学村": "Jimei School Village",
    "沙坡尾": "Shapowei",
    "曾厝垵": "Zengcuo'an",
    "黄厝": "Huangcuo",
    "白城": "Baicheng",
    "五缘湾": "Wuyuan Bay",
    "筼筜湖": "Yundang Lake",
    "员筜湖": "Yundang Lake",
    "环岛路": "Island Ring Road",
    "陈嘉庚": "Tan Kah Kee",
    "张万森": "Zhang Wansen",
    "林北星": "Lin Beixing",
    "葛优躺": "Ge You Reclining Spot",
    "双子塔": "Twin Towers",
    "花鸟市场": "Flower and Bird Market",
    "第八市场": "Eighth Market",
    "八市": "Eighth Market",
    "老城区": "Old Town",
    "市中心": "City Center",
    "海边": "Seafront",
    "日出": "Sunrise",
    "日落": "Sunset",
    "看海": "Sea View",
    "看景": "Scenic View",
    "观景": "Scenic View",
    "爬山": "Hiking",
    "拍照": "Photography",
    "打卡": "Landmark Visit",
    "美食": "Food",
    "小吃": "Snacks",
    "咖啡": "Coffee",
    "下午茶": "Afternoon Tea",
    "文创": "Creative Arts",
    "文艺": "Arts",
    "入口": "Entrance",
    "出口": "Exit",
    "起点": "Starting Point",
    "终点": "End Point",
    "附近": "Nearby",
    "旁边": "Beside",
    "对面": "Opposite",
    "导航": "Navigate to",
    "路线": "Route",
    "门口": "Entrance",
    "老店": "Original Shop",
    "总店": "Flagship Shop",
    "分店": "Branch",
    "店内": "Inside the Shop",
    "楼顶": "Rooftop",
    "天台": "Rooftop",
    "一楼": "First Floor",
    "二楼": "Second Floor",
    "三楼": "Third Floor",
    "看日落": "Sunset View",
    "看日出": "Sunrise View",
    "花海": "Flower Field",
    "晚餐": "Dinner",
    "外观": "Exterior",
    "沿海": "Coastal",
    "骑行": "Cycling",
    "租车": "Bike Rental",
    "墙": "Wall",
}

FACILITY_SUFFIXES = {
    "国际邮轮中心": "International Cruise Center",
    "国际会展中心": "International Conference and Exhibition Center",
    "游客服务中心": "Visitor Service Center",
    "游客中心": "Visitor Center",
    "文化艺术中心": "Culture and Arts Center",
    "购物中心": "Shopping Center",
    "商业中心": "Commercial Center",
    "体育中心": "Sports Center",
    "服务中心": "Service Center",
    "艺术西区": "Art Zone",
    "创意园区": "Creative District",
    "文化创意园": "Cultural and Creative Park",
    "科技产业园": "Technology Park",
    "工业园区": "Industrial Park",
    "湿地公园": "Wetland Park",
    "森林公园": "Forest Park",
    "郊野公园": "Country Park",
    "文化公园": "Cultural Park",
    "儿童公园": "Children's Park",
    "地质公园": "Geological Park",
    "植物园": "Botanical Garden",
    "公园": "Park",
    "花鸟市场": "Flower and Bird Market",
    "农贸市场": "Farmers' Market",
    "菜市场": "Produce Market",
    "海鲜市场": "Seafood Market",
    "市场": "Market",
    "地铁站": "Metro Station",
    "地铁口": "Metro Entrance",
    "公交车站": "Bus Stop",
    "公交站": "Bus Stop",
    "火车站": "Railway Station",
    "动车站": "Railway Station",
    "高铁站": "High-Speed Railway Station",
    "客运站": "Coach Station",
    "机场": "Airport",
    "轮渡码头": "Ferry Terminal",
    "客运码头": "Passenger Ferry Terminal",
    "游艇码头": "Marina",
    "码头": "Pier",
    "渡口": "Ferry Landing",
    "步行街": "Pedestrian Street",
    "商业街": "Commercial Street",
    "美食街": "Food Street",
    "小吃街": "Snack Street",
    "咖啡街": "Coffee Street",
    "老街": "Old Street",
    "街道": "Subdistrict",
    "大道": "Avenue",
    "公路": "Highway",
    "街": "Street",
    "路": "Road",
    "巷": "Lane",
    "弄": "Alley",
    "社区": "Community",
    "小区": "Residential Community",
    "村落": "Village",
    "村": "Village",
    "镇": "Town",
    "广场": "Plaza",
    "花园": "Garden",
    "学村": "School Village",
    "大学": "University",
    "学院": "College",
    "中学": "Middle School",
    "小学": "Primary School",
    "学校": "School",
    "幼儿园": "Kindergarten",
    "艺术博物馆": "Art Museum",
    "美术馆": "Art Museum",
    "博物馆": "Museum",
    "纪念馆": "Memorial Museum",
    "艺术馆": "Art Gallery",
    "展览馆": "Exhibition Hall",
    "图书馆": "Library",
    "书店": "Bookstore",
    "电影院": "Cinema",
    "剧院": "Theatre",
    "戏院": "Theatre",
    "音乐厅": "Concert Hall",
    "体育馆": "Sports Hall",
    "体育场": "Stadium",
    "故居": "Former Residence",
    "旧址": "Historic Site",
    "遗址": "Heritage Site",
    "祖庙": "Ancestral Temple",
    "祠堂": "Ancestral Hall",
    "教堂": "Church",
    "寺庙": "Temple",
    "禅寺": "Buddhist Temple",
    "寺": "Temple",
    "庙": "Temple",
    "宫": "Temple",
    "祠": "Ancestral Hall",
    "炮台": "Fortress",
    "城堡": "Castle",
    "城墙": "City Wall",
    "灯塔": "Lighthouse",
    "索道": "Cableway",
    "观景平台": "Viewing Platform",
    "观景台": "Viewpoint",
    "观景点": "Viewpoint",
    "拍照点": "Photo Spot",
    "打卡点": "Landmark",
    "健康步道": "Health Trail",
    "木栈道": "Boardwalk",
    "栈道": "Boardwalk",
    "步道": "Trail",
    "绿道": "Greenway",
    "长堤": "Long Causeway",
    "大桥": "Bridge",
    "天桥": "Footbridge",
    "桥": "Bridge",
    "隧道": "Tunnel",
    "水库": "Reservoir",
    "海湾": "Bay",
    "沙滩": "Beach",
    "海滩": "Beach",
    "海滨": "Seafront",
    "海岸": "Coast",
    "湖": "Lake",
    "池": "Pond",
    "溪": "Creek",
    "河": "River",
    "岛": "Island",
    "礁": "Reef",
    "山峰": "Peak",
    "山": "Mountain",
    "峰": "Peak",
    "岭": "Ridge",
    "岩": "Rock",
    "洞": "Cave",
    "别墅": "Villa",
    "老宅": "Historic Residence",
    "公馆": "Mansion",
    "大楼": "Building",
    "大厦": "Tower",
    "公寓": "Apartments",
    "酒店": "Hotel",
    "宾馆": "Hotel",
    "民宿": "Homestay",
    "客栈": "Inn",
    "咖啡馆": "Café",
    "咖啡店": "Café",
    "面包店": "Bakery",
    "甜品店": "Dessert Shop",
    "奶茶店": "Tea Shop",
    "便利店": "Convenience Store",
    "杂货铺": "General Store",
    "超市": "Supermarket",
    "购物城": "Shopping Center",
    "商场": "Shopping Mall",
    "大排档": "Food Stall",
    "排档": "Food Stall",
    "小吃店": "Snack Shop",
    "餐厅": "Restaurant",
    "餐馆": "Restaurant",
    "饭店": "Restaurant",
    "菜馆": "Restaurant",
    "茶馆": "Teahouse",
    "茶室": "Tea Room",
    "酒吧": "Bar",
    "商店": "Shop",
    "店铺": "Shop",
    "店": "Shop",
    "工厂": "Factory",
    "仓库": "Warehouse",
    "园区": "District",
    "园": "Garden",
    "中心": "Center",
    "段": "Section",
    "站": "Station",
}

WHOLE_TERMS = {
    "东门": "East Gate",
    "西门": "West Gate",
    "南门": "South Gate",
    "北门": "North Gate",
    "大门": "Main Gate",
    "入口": "Entrance",
    "出口": "Exit",
    "门口": "Entrance",
    "海边": "Seafront",
    "海上列车": "Sea Train",
    "轮渡": "Ferry",
    "老城区": "Old Town",
    "市中心": "City Center",
    "地下通道": "Underpass",
    "天台": "Rooftop",
    "楼顶": "Rooftop",
    "下沉广场": "Sunken Plaza",
}


def clean_source_name(value: str) -> str:
    text = value.strip()
    text = text.replace("[话题]", "").replace("【话题】", "")
    text = re.sub(r"#+", " ", text)
    text = text.replace("（", " (").replace("）", ") ")
    text = text.replace("：", ": ").replace("，", ", ").replace("、", ", ")
    text = text.replace("—", " – ").replace("——", " – ")
    text = re.sub(r"[?？]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;·")
    return text


def romanize(text: str) -> str:
    syllables = lazy_pinyin(text, errors=lambda value: list(value))
    joined = "".join(syllable for syllable in syllables if syllable.strip())
    return joined[:1].upper() + joined[1:]


def translate_chinese_span(span: str) -> str:
    if span in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[span]
    if span in WHOLE_TERMS:
        return WHOLE_TERMS[span]
    if span in PHRASE_TRANSLATIONS:
        return PHRASE_TRANSLATIONS[span]

    for suffix in sorted(FACILITY_SUFFIXES, key=len, reverse=True):
        if span.endswith(suffix) and len(span) > len(suffix):
            stem = span[: -len(suffix)]
            stem_label = translate_chinese_span(stem)
            return f"{stem_label} {FACILITY_SUFFIXES[suffix]}".strip()
        if span == suffix:
            return FACILITY_SUFFIXES[suffix]

    for phrase in sorted(PHRASE_TRANSLATIONS, key=len, reverse=True):
        if span.startswith(phrase) and len(span) > len(phrase):
            remainder = span[len(phrase) :]
            return f"{PHRASE_TRANSLATIONS[phrase]} {translate_chinese_span(remainder)}".strip()

    for phrase in sorted(PHRASE_TRANSLATIONS, key=len, reverse=True):
        if phrase in span:
            before, after = span.split(phrase, 1)
            parts = [translate_chinese_span(before) if before else "", PHRASE_TRANSLATIONS[phrase]]
            if after:
                parts.append(translate_chinese_span(after))
            return " ".join(part for part in parts if part).strip()

    return romanize(span)


def translate_name(source_name: str) -> tuple[str, str]:
    source = source_name.strip()
    cleaned = clean_source_name(source)
    exact_key = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip()
    if source in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[source], "manual"
    if cleaned in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[cleaned], "manual"
    if exact_key in EXACT_TRANSLATIONS and exact_key != cleaned:
        suffix = cleaned[len(exact_key) :].strip()
        translated_suffix = CHINESE_RUN.sub(lambda match: translate_chinese_span(match.group()), suffix)
        return f"{EXACT_TRANSLATIONS[exact_key]} {translated_suffix}".strip(), "manual+rule"
    if not CHINESE.search(cleaned):
        return cleaned or "Unnamed Place", "existing-English"

    translated = cleaned
    for phrase in sorted(EXACT_TRANSLATIONS, key=len, reverse=True):
        if phrase in translated:
            translated = translated.replace(phrase, f" {EXACT_TRANSLATIONS[phrase]} ")
    translated = CHINESE_RUN.sub(lambda match: f" {translate_chinese_span(match.group())} ", translated)
    translated = translated.replace("( ", "(").replace(" )", ")")
    translated = re.sub(r"\s+([,;:)])", r"\1", translated)
    translated = re.sub(r"([(])\s+", r"\1", translated)
    translated = re.sub(r"\s+", " ", translated).strip(" ,;·–-")
    translated = re.sub(r"\bNo\s+(\d+)\b", r"No. \1", translated)
    translated = re.sub(r"(.+? Road)\s+(\d+)\s+Hao\b", r"No. \2 \1", translated)
    return translated or romanize(source), "cartographic-rule"


def collect_names() -> list[str]:
    names: set[str] = set()
    for path, column in [(NODES_CSV, "location_POI"), (RECORDS_CSV, "location_POI")]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get(column, "")).strip()
                if name:
                    names.add(name)
    return sorted(names)


def main() -> None:
    names = collect_names()
    rows = []
    for name in names:
        english, method = translate_name(name)
        if CHINESE.search(english):
            raise ValueError(f"Untranslated Chinese remains: {name!r} -> {english!r}")
        rows.append({"name_zh": name, "name_en": english, "method": method})
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name_zh", "name_en", "method"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {OUTPUT_CSV}")
    print(f"Translated {len(rows):,} unique place names")


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9 or newer is required")
    main()
