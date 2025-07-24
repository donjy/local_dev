import json
import ccxt


def test_binance_connection():
    # 读取配置文件
    
    with open('user_data/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    ccxt_config = config['exchange'].get('ccxt_config', {})
    # 初始化 binance 实例
    print(ccxt_config.get('httpsProxy'))
    exchange = ccxt.binance({
        'timeout': ccxt_config.get('timeout', 10000),
        'httpsProxy': ccxt_config.get('httpsProxy'),
        'wsProxy': ccxt_config.get('wsProxy'),
        'enableRateLimit': True
    })
    try:
        markets = exchange.public_get_exchangeinfo()
        print('连接成功，返回数据长度:', len(markets))
    except Exception as e:
        print('连接失败:', e)


def foo():
    test_binance_connection()


if __name__ == '__main__':
    foo()