import pymysql


def start_engine():
    config = {
        'host': 'wqdcsrv3396.cn.infra',
        'port': 3313,
        'user': 'app_risk',
        'password': 'uiwe8_k34kJSKJdwd',
        'database': 'db_risk',
    }

    return pymysql.connect(**config)


def save_conversation(product_name):
    from nanoid import generate

    conversation_id = generate()
    insert_query = '''
    INSERT INTO riskchat_conversations (conversation_id, product_name)
    VALUES (%s, %s)
    '''
    try:
        with start_engine() as engine:
            with engine.cursor() as cursor:
                cursor.execute(insert_query, (conversation_id, product_name))
                engine.commit()
        return conversation_id
    except pymysql.MySQLError as e:
        raise e



def save_message(convo_id, human, ai, agents):
    from nanoid import generate

    msg_id = generate()
    insert_query = '''
    INSERT INTO riskchat_messages (message_id, conversation_id, human, ai, agents)
    VALUES (%s, %s, %s, %s, %s)
    '''
    try:
        with start_engine() as engine:
            with engine.cursor() as cursor:
                cursor.execute(insert_query, (msg_id, convo_id, human, ai, agents))
                engine.commit()
    except pymysql.MySQLError as e:
        raise e
