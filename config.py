import yaml

from addict import Dict


class Config:
    _instance = None

    class SafeLoaderWithJoin(yaml.SafeLoader):
        pass

    @classmethod
    def join_constructor(cls, loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def __new__(cls, path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.config = None
            if path:
                cls._instance._load_config(path)
        return cls._instance

    def _load_config(self, path):
        self.SafeLoaderWithJoin.add_constructor('!join', self.join_constructor)
        # 打开YAML文件
        with open(path, 'r') as file:
            # 加载YAML内容
            config = yaml.load(file, self.SafeLoaderWithJoin)

        # 现在config是一个包含YAML内容的Python字典
        self.config = Dict(config)

    def get_config(self):
        return self.config

    def __getattr__(self, name):
        return getattr(self.config, name)


if __name__ == '__main__':
    config = Config('example_fuse.yml')
    c = config.get_config()
    print(c)
