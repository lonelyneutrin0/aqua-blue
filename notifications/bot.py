import argparse
import asyncio

import discord
import aqua_blue


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("key", type=str)
    parser.add_argument("channel", type=int)
    args = parser.parse_args()

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        await client.wait_until_ready()
        channel = client.get_channel(args.channel)
        embed = discord.Embed(
            title=f"🌊 aqua-blue {aqua_blue.__version__} Released! 🚀",
            url="https://pypi.org/project/aqua-blue/",
            description="A new version of aqua-blue is now available on PyPI!",
            color=0x3498db
        )
        embed.set_footer(text="Check it out now!")

        await channel.send(embed=embed)
        await client.close()

    client.run(args.key)


if __name__ == "__main__":

    main()
