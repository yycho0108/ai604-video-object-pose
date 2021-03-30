#!/usr/bin/env python3

from google.cloud import storage


def glob_objectron():
    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs('objectron',
                              prefix='v1/records_shuffled/cup/cup_train')
    for blob in blobs:
        print(str(blob))


def main():
    glob_objectron()
    pass


if __name__ == '__main__':
    main()
