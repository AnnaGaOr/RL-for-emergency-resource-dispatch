# About OpenRouteService, and usage of the API

OpenRouteService is a service that offers multiple APIs for routing, geocoding, and other services. It is free to use.
We use the API for estimation of time that is required for an ambulance to reach the site of the accident, through
the `OpenRouteTravelTimeModel`.

## Usage

The `OpenRouteTravelTimeModel` can be used by creating an instance of it, with proper parameters. If the server is
hosted in the lab, there is no need to pass any parameters. If you want to use the online service (limited to 2500
requests per day), you need to pass `mode='online'`. This mode also requires you to pass `api_key=YOUR_API_KEY`. You can
get your own API key from [here](https://openrouteservice.org/dev/#/home). If you want to use the custom server instead,
hosted somewhere else, you need to pass `api_url=YOUR_URL` and `api_key=YOUR_API_KEY`.

## Hosting the service on the lab's server

The lab's server is equipped with a local OpenRouteService, with the map of Switzerland. It is located in the root
directory, in the `openrouteservice` folder. Instructions to run it are as follows:

1. Keep spamming `cd ..` until you reach the root directory.
2. `cd openrouteservice`
3. `cd docker`
4. `sudo docker-compose up`
5. Enter your password. If you're not an administrator, ask somebody at the lab about it.
6. After the server starts, you can use the API.
7. If you need to do something else on the server, press `Ctrl+Z` to suspend the service, and then `bg` to have it
   continue in the background. Read more about running in
   background [here](https://www.thegeekdiary.com/understanding-the-job-control-commands-in-linux-bg-fg-and-ctrlz/).
